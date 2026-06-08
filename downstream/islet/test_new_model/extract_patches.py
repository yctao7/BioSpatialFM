"""
Extract islet patches from TIFF images using islet-level CSV metadata.

Output:
    output/all_patches_islet_csv/<patient_id>/islet_<id>.npy
    output/all_patches_islet_csv/<patient_id>/channels.json

Cropping rule (from islets.csv):
  1. Center on (Centroid X um, Centroid Y um)
  2. Side length = Max diameter um
  3. Crop a square patch (clamped to image bounds)

Pixel conversion uses per-image resolution from BioSpatialFM/resolution.txt.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import argparse
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import yaml
import torch
import torch.nn.functional as F


OUTPUT_DIR = Path(__file__).parent / "output" / "all_patches_islet_csv"


# ---------------------------------------------------------------------------
# TIFF loading + channel name extraction
# ---------------------------------------------------------------------------

def get_channel_names_qptiff(tif):
    """Extract channel names from PerkinElmer qptiff (per-page XML)."""
    names = []
    for page in tif.pages:
        desc = page.description
        if not desc:
            continue
        try:
            root = ET.fromstring(desc)
        except ET.ParseError:
            continue
        if root.findtext("ImageType") == "FullResolution":
            name = root.findtext("Name")
            if name:
                names.append(name)
    return names


def get_channel_names_ometiff(tif):
    """Extract channel names from OME-TIFF XML metadata."""
    if not tif.ome_metadata:
        return []
    try:
        root = ET.fromstring(tif.ome_metadata)
    except ET.ParseError:
        return []
    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    channels = root.findall(".//ome:Channel", ns)
    return [ch.get("Name", f"ch{i}") for i, ch in enumerate(channels)]


def load_tiff_with_channels(file_path):
    """Load multiplex TIFF and return (image[C,H,W], channel_names)."""
    print(f"  Loading TIFF: {Path(file_path).name}")
    with tifffile.TiffFile(file_path) as tif:
        if tif.ome_metadata:
            channel_names = get_channel_names_ometiff(tif)
        else:
            channel_names = get_channel_names_qptiff(tif)
        data = tif.asarray()

    if data.ndim == 3 and data.shape[0] > data.shape[2]:
        data = np.transpose(data, (2, 0, 1))
    elif data.ndim == 4:
        data = data[0]

    n_ch = data.shape[0]
    print(f"  Image shape: {data.shape}  ({n_ch} channels)")

    if len(channel_names) < n_ch:
        print(
            f"  WARNING: only {len(channel_names)} channel names found, "
            f"padding remaining {n_ch - len(channel_names)} as 'unknown_<i>'"
        )
        channel_names += [f"unknown_{i}" for i in range(len(channel_names), n_ch)]
    elif len(channel_names) > n_ch:
        channel_names = channel_names[:n_ch]

    print(f"  Channels: {channel_names}")
    return data, channel_names


# ---------------------------------------------------------------------------
# Islet metadata + resolution parsing
# ---------------------------------------------------------------------------

def load_islet_metadata(islet_csv_path):
    """Load and clean islet-level metadata CSV."""
    print(f"Loading islet metadata CSV: {Path(islet_csv_path).name}")
    df = pd.read_csv(islet_csv_path)

    required = [
        "Image",
        "Classification",
        "Centroid X µm",
        "Centroid Y µm",
        "Max diameter µm",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"islet metadata missing columns: {missing}")

    df = df[df["Classification"].astype(str).str.upper().str.contains("ISLET", na=False)].copy()
    for c in ["Centroid X µm", "Centroid Y µm", "Max diameter µm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Centroid X µm", "Centroid Y µm", "Max diameter µm"])

    if "Circularity" in df.columns:
        df["Circularity"] = pd.to_numeric(df["Circularity"], errors="coerce")

    print(f"  Loaded {len(df)} valid islet rows")
    return df


def _norm_text(s):
    return re.sub(r"[^A-Z0-9]+", "", str(s).upper())


def load_resolution_map(resolution_txt_path):
    """Parse resolution.txt to map image path/name -> (um_per_px_x, um_per_px_y)."""
    print(f"Loading resolution map: {Path(resolution_txt_path).name}")
    path_map = {}
    base_map = {}
    stem_map = {}

    with open(resolution_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "ERROR" in line.upper():
                continue

            # Robust parsing: path may contain spaces (e.g., "CODEX Staining").
            m = re.match(r"^(.*\S)\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)\s*$", line)
            if not m:
                continue
            rel = m.group(1)
            try:
                mpp_x = float(m.group(2))
                mpp_y = float(m.group(3))
            except ValueError:
                continue

            p = Path(rel)
            key_path = _norm_text(rel)
            key_base = _norm_text(p.name)
            key_stem = _norm_text(p.stem)

            path_map[key_path] = (mpp_x, mpp_y)
            base_map[key_base] = (mpp_x, mpp_y)
            stem_map[key_stem] = (mpp_x, mpp_y)

    print(f"  Parsed {len(path_map)} resolution rows")
    return {
        "path": path_map,
        "base": base_map,
        "stem": stem_map,
    }


def get_image_resolution_mpp(image_path, res_map):
    """Find per-image resolution (um/px) by path/name matching."""
    p = Path(image_path)
    key_path = _norm_text(str(image_path))
    key_base = _norm_text(p.name)
    key_stem = _norm_text(p.stem)

    if key_path in res_map["path"]:
        return res_map["path"][key_path]
    if key_base in res_map["base"]:
        return res_map["base"][key_base]
    if key_stem in res_map["stem"]:
        return res_map["stem"][key_stem]

    return None


def get_patient_islets_from_metadata(islet_df, patient_info, min_circularity=0.05):
    """Select islet rows that belong to this patient/image.

    Args:
        min_circularity: Islets with Circularity below this threshold are skipped.
            These are typically annotation artifacts where a large irregular region
            was mis-labeled as a single islet (e.g. Circularity < 0.05 while the
            normal islet median is ~0.24). Set to 0 to disable.
    """
    pid = patient_info["patient_id"]
    img_path = patient_info["image_path"]
    islet_image_name = patient_info.get("islet_image_name", None)

    pid_token = _norm_text(pid).replace("-", "")
    basename = _norm_text(Path(img_path).name)
    stem = _norm_text(Path(img_path).stem)

    work = islet_df.copy()
    img_norm = work["Image"].astype(str).map(_norm_text)

    # Prefer exact image-name matching when config provides first-column Image string.
    if islet_image_name:
        key = _norm_text(islet_image_name)
        work = work[img_norm == key].copy()
    else:
        mask = (
            img_norm.str.contains(pid_token, na=False)
            | img_norm.str.contains(stem, na=False)
            | img_norm.str.contains(basename, na=False)
        )
        work = work[mask].copy()

    if work.empty:
        return []

    # Filter out annotation artifacts with extremely low circularity.
    has_circularity = "Circularity" in work.columns and min_circularity > 0
    if has_circularity:
        before = len(work)
        bad = work["Circularity"].notna() & (work["Circularity"] < min_circularity)
        if bad.any():
            skipped = work[bad][["Name", "Max diameter µm", "Circularity"]]
            for _, r in skipped.iterrows():
                print(
                    f"  SKIP {r['Name']}: Circularity={r['Circularity']:.4f} < {min_circularity}"
                    f"  (Max diameter={r['Max diameter µm']:.0f} µm)"
                )
        work = work[~bad].copy()
        if len(work) < before:
            print(f"  Filtered {before - len(work)} low-circularity islets, {len(work)} remain")

    rows = []
    for i, row in work.iterrows():
        name = str(row.get("Name", ""))
        m = re.search(r"(\d+)", name)
        islet_id = int(m.group(1)) if m else int(i)
        rows.append(
            {
                "islet_id": islet_id,
                "center_x_um": float(row["Centroid X µm"]),
                "center_y_um": float(row["Centroid Y µm"]),
                "diameter_um": float(row["Max diameter µm"]),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Patch cropping + saving
# ---------------------------------------------------------------------------

def _square_bounds_center_fixed(cx, cy, side):
    """Build target square bounds centered at (cx, cy), without shifting."""
    side = max(int(round(side)), 16)
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = x1 + side
    y2 = y1 + side
    return x1, y1, x2, y2, side


def _resize_patch_to_square(patch, side):
    """Resize patch [C,H,W] to [C,side,side] using bilinear interpolation."""
    if patch.shape[1] == side and patch.shape[2] == side:
        return patch
    x = torch.from_numpy(patch).unsqueeze(0).to(torch.float32)
    x = F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False)
    out = x.squeeze(0).cpu().numpy()
    return out


def crop_patches_from_islet_metadata(image, islet_rows_um, mpp_x, mpp_y):
    """Center-fixed square crop; if out-of-bounds, interpolate visible crop to square."""
    _, H, W = image.shape
    patches, meta_list = [], []

    # Use axis-specific conversion for center, and average scale for diameter.
    mpp_d = (float(mpp_x) + float(mpp_y)) / 2.0

    for r in islet_rows_um:
        cx = r["center_x_um"] / float(mpp_x)
        cy = r["center_y_um"] / float(mpp_y)
        side = r["diameter_um"] / float(mpp_d)

        # Target box is fixed at center; no boundary-shift.
        tx1, ty1, tx2, ty2, target_side = _square_bounds_center_fixed(cx, cy, side)
        # Clip to image bounds for actual read.
        x1 = max(0, tx1)
        y1 = max(0, ty1)
        x2 = min(W, tx2)
        y2 = min(H, ty2)
        ph, pw = y2 - y1, x2 - x1
        if ph < 2 or pw < 2:
            continue

        patch = image[:, y1:y2, x1:x2]
        patch = _resize_patch_to_square(patch, target_side)
        patches.append(patch)
        meta_list.append(
            {
                "islet_id": int(r["islet_id"]),
                "y": y1,
                "x": x1,
                "patch_h": int(target_side),
                "patch_w": int(target_side),
                "center_x_px": cx,
                "center_y_px": cy,
                "diameter_px": float(target_side),
                "mpp_x": float(mpp_x),
                "mpp_y": float(mpp_y),
                "was_clipped": bool(tx1 < 0 or ty1 < 0 or tx2 > W or ty2 > H),
            }
        )

    return patches, meta_list


def save_patches(patches, meta_list, channel_names, patient_id, output_dir):
    """Save patches and channels.json under patient folder."""
    patient_dir = Path(output_dir) / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    for patch, meta in zip(patches, meta_list):
        np.save(patient_dir / f"islet_{meta['islet_id']}.npy", patch)

    with open(patient_dir / "channels.json", "w") as f:
        json.dump(channel_names, f, indent=2)

    print(f"  Saved {len(patches)} patches + channels.json -> {patient_dir}/")


# ---------------------------------------------------------------------------
# Per-patient pipeline
# ---------------------------------------------------------------------------

def process_patient(patient_info, output_dir, islet_df, res_map, fallback_mpp=None, min_circularity=0.05):
    t0 = time.time()
    pid = patient_info["patient_id"]
    label = patient_info.get("label", "unknown")
    img_path = patient_info["image_path"]

    print(f"\n{'=' * 55}")
    print(f"  {pid}  ({label})")
    print(f"{'=' * 55}")

    if not os.path.exists(img_path):
        print(f"  SKIP - image not found: {img_path}")
        return

    patient_dir = Path(output_dir) / pid
    if patient_dir.exists() and any(patient_dir.glob("islet_*.npy")) and (patient_dir / "channels.json").exists():
        print(f"  SKIP - patches already exist in {patient_dir}/")
        return

    try:
        image, channel_names = load_tiff_with_channels(img_path)
    except Exception as e:
        print(f"  SKIP - failed to read TIFF: {Path(img_path).name}")
        print(f"         reason: {e}")
        return

    mpp = get_image_resolution_mpp(img_path, res_map)
    if mpp is None:
        if fallback_mpp is None:
            print(f"  SKIP - no resolution match in resolution.txt for {Path(img_path).name}")
            del image
            return
        mpp = (float(fallback_mpp), float(fallback_mpp))
        print(f"  WARNING: no resolution match, fallback mpp={fallback_mpp}")

    mpp_x, mpp_y = mpp
    print(f"  Resolution: mpp_x={mpp_x:.6f}, mpp_y={mpp_y:.6f}")

    islet_rows_um = get_patient_islets_from_metadata(islet_df, patient_info, min_circularity=min_circularity)
    print(f"  Islet metadata rows matched: {len(islet_rows_um)}")
    if not islet_rows_um:
        print("  SKIP - no matching islets in islet metadata CSV")
        del image
        return

    patches, meta_list = crop_patches_from_islet_metadata(image, islet_rows_um, mpp_x, mpp_y)
    print(f"  Cropped {len(patches)} islet patches")
    if not patches:
        del image
        return

    save_patches(patches, meta_list, channel_names, pid, output_dir)
    del image, patches

    print(f"  Done in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract islet patches from islets.csv + resolution.txt (no GPU needed)."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--start-from",
        default=None,
        help="Start processing from this patient_id (inclusive), e.g. HPAP-108 or HPAP-153-img2",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Process only this single patient_id, e.g. HPAP-146",
    )
    parser.add_argument(
        "--min-circularity",
        type=float,
        default=0.05,
        help="Skip islets with Circularity below this threshold (default: 0.05). "
             "Normal islets have median ~0.24; values below 0.05 are annotation artifacts. "
             "Set to 0 to disable filtering.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config.get("output_dir", OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)

    islet_csv_path = config.get(
        "islet_metadata_csv",
        "/nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/CODEX/export_metadata/islets.csv",
    )
    islet_df = load_islet_metadata(islet_csv_path)

    resolution_txt_path = config.get(
        "resolution_txt",
        str(Path(__file__).resolve().parents[3] / "resolution.txt"),
    )
    res_map = load_resolution_map(resolution_txt_path)

    fallback_mpp = config.get("fallback_microns_per_pixel", None)

    patients = config.get("patients", [])
    patients = [p for p in patients if "image_path" in p]
    if args.only:
        patients = [p for p in patients if p.get("patient_id") == args.only]
        if not patients:
            raise ValueError(f"--only not found in config patients: {args.only}")
    elif args.start_from:
        start_idx = None
        for i, p in enumerate(patients):
            if p.get("patient_id") == args.start_from:
                start_idx = i
                break
        if start_idx is None:
            raise ValueError(f"--start-from not found in config patients: {args.start_from}")
        patients = patients[start_idx:]

    labels = sorted(set(p.get("label", "?") for p in patients))

    print("=" * 55)
    print("Islet Patch Extraction [CSV + resolution.txt]")
    print(f"  Config          : {config_path}")
    print(f"  Output          : {output_dir}")
    print(f"  Islet CSV       : {islet_csv_path}")
    print(f"  Resolution      : {resolution_txt_path}")
    print(f"  Patients        : {len(patients)}  (labels: {labels})")
    print(f"  Min circularity : {args.min_circularity} ({'disabled' if args.min_circularity == 0 else 'filtering annotation artifacts'})")
    print("=" * 55)

    for idx, pinfo in enumerate(patients, 1):
        print(f"\n[{idx}/{len(patients)}]", end="")
        process_patient(
            pinfo,
            output_dir=output_dir,
            islet_df=islet_df,
            res_map=res_map,
            fallback_mpp=fallback_mpp,
            min_circularity=args.min_circularity,
        )

    all_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    with_channels = [d for d in all_dirs if (d / "channels.json").exists()]
    print(f"\n{'=' * 55}")
    print(f"Done. {len(with_channels)} patients have patches + channels.json")
    print(f"Output -> {output_dir}/")


if __name__ == "__main__":
    main()
