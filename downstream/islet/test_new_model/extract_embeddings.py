"""
Extract islet embeddings for a given checkpoint and save to
    output/<run_name>/embeddings.npy
    output/<run_name>/metadata.csv

Usage:
    python extract_embeddings.py
    python extract_embeddings.py --checkpoint /path/to/checkpoint.pth --run_name my_experiment
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import sys
import argparse
import time
import json
import os
import math
import types
import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# xFormers memory-efficient attention is CUDA-only in this environment.
# Disable it automatically for CPU inference before importing kronos modules.
if not torch.cuda.is_available():
    os.environ.setdefault("XFORMERS_DISABLED", "1")

sys.path.append(str(Path(__file__).parent.parent))

from kronos.inference import create_model


# ---------------------------------------------------------------------------
# Panel marker orders (channel-index -> marker semantic)
# ---------------------------------------------------------------------------

# Canonical marker ordering for known CODEX panels.
# Used as a safe fallback when channels.json stores only dye names (e.g. FITC/Cy5).
PANEL_MARKERS_BY_CHANNEL_COUNT = {
    25: [
        "DAPI","GCG","CD45","CD19","ECAD","COL1A1","LAM","SST","KRT","DPP4",
        "ACTA2","CPEP","CD3","CD31","GHRL","CD11C","LYVE1","MCAM","IBA1","CD117",
        "PROINS","COL4A1","PNLIP","PPY","SYP",
    ],
    55: [
        "DAPI","GCG","CD45","CD19","IRX2","CD8","LAM","PDX1","NEUROD1","CD3",
        "SST","HLA-DR","CD4","ACTA2","COL1A1","VIM","PAX6","PPY","CD11C","NKX6-1",
        "KRT","ARX","EPCAM","SELP","CD31","HSPG2","CD117","NKX2-2","GP2","CDX2",
        "GHRL","MMR","ATP1A1","FN1","PD-L1","MPO","MCAM","CHGA","SYP","CD163",
        "CD68","COL6","NPY","COL4A1","CD66B","CD44","PNLIP","CD141","CD90","CD39L3",
        "SOX9","CPEP","KI67","TUBB3","LYVE1",
    ],
    56: [
        "DAPI","GCG","CD45","CD19","IRX2","CD8","LAM","PDX1","NEUROD1","CD3",
        "SST","CD40","CD4","ACTA2","COL1A1","VIM","PAX6","PPY","CD11C","NKX6-1",
        "KRT","ARX","EPCAM","SELP","HLA-DR","CD31","HSPG2","CD117","NKX2-2","GP2",
        "CDX2","GHRL","MMR","ATP1A1","FN1","PD-L1","MPO","MCAM","CHGA","SYP",
        "CD163","CD68","COL6","NPY","COL4A1","CD66B","CD44","PNLIP","CD141","CD90",
        "CD39L3","SOX9","CPEP","KI67","TUBB3","LYVE1",
    ],
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path, model_type="vits16"):
    """Load KRONOS backbone from a DINO checkpoint with training-compatible config.

    Handles teacher/student/model key variants and strips common
    state-dict prefixes. Prefers constructing the model with the same
    architecture flags used during training (token overlap, drop path, etc.).
    Returns (model, device, embedding_dim).
    """
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "teacher" in checkpoint:
        state_dict = checkpoint["teacher"]
        print("  Using 'teacher' weights")
    elif "student" in checkpoint:
        state_dict = checkpoint["student"]
        print("  Using 'student' weights")
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
        print("  Using 'model' weights")
    else:
        state_dict = checkpoint

    # Build inference model using training-time architecture when available.
    ckpt_args = checkpoint.get("args", None) if isinstance(checkpoint, dict) else None
    if ckpt_args is not None:
        if hasattr(ckpt_args, "__dict__"):
            ckpt_args = vars(ckpt_args)
        elif not isinstance(ckpt_args, dict):
            ckpt_args = None

    cfg = {
        "model_type": (ckpt_args.get("model_type") if ckpt_args else model_type) or model_type,
        "token_overlap": bool(ckpt_args.get("token_overlap", False)) if ckpt_args else False,
        "drop_path_rate": float(ckpt_args.get("drop_path_rate", 0.1)) if ckpt_args else 0.1,
    }
    model, _, embedding_dim = create_model(cfg=cfg)
    print(f"  Model cfg: {cfg}")

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        # Keep only backbone weights when checkpoint contains DINO heads.
        if new_key.startswith("backbone."):
            new_key = new_key[len("backbone."):]
        elif new_key.startswith("head.") or "dino_head" in new_key:
            continue

        for prefix in ["module.", "encoder.", "base_encoder."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        new_state_dict[new_key] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"  Missing keys: {len(msg.missing_keys)}  Unexpected: {len(msg.unexpected_keys)}")

    # Patch positional interpolation to support rectangular token grids.
    # Upstream implementation assumes square grids in some code paths.
    def _rect_interpolate_pos_encoding(self, x, w, h, npatch):
        previous_dtype = x.dtype
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]

        patch_size = self.patch_size[0] if isinstance(self.patch_size, tuple) else self.patch_size
        stride_size = self.stride_size[0] if isinstance(self.stride_size, tuple) else self.stride_size
        h_tokens = len(range(0, h - patch_size + 1, stride_size))
        w_tokens = len(range(0, w - patch_size + 1, stride_size))

        # Fallback: infer a near-square grid only if geometry-based counting fails.
        if h_tokens * w_tokens != npatch:
            h_tokens = int(round(math.sqrt(npatch)))
            h_tokens = max(h_tokens, 1)
            w_tokens = max(npatch // h_tokens, 1)
            while h_tokens * w_tokens < npatch:
                w_tokens += 1

        M = int(math.sqrt(N))
        assert N == M * M
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            size=(h_tokens, w_tokens),
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    model.interpolate_pos_encoding = types.MethodType(_rect_interpolate_pos_encoding, model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"  Embedding dim: {embedding_dim}  Device: {device}")
    return model, device, embedding_dim


def load_marker_metadata(csv_path):
    """Load per-channel normalization stats and semantic marker IDs from a marker metadata CSV.

    The CSV must have columns: channel_id, marker_name, marker_mean, marker_std, marker_id.
      - channel_id  : position of the channel in the image (0-based)
      - marker_id   : semantic ID used in the model's marker embedding table during training

    Returns:
        df           : full DataFrame sorted by channel_id
        means        : float array of length n_channels, means[channel_id] = marker_mean
        stds         : float array of length n_channels, stds[channel_id]  = marker_std
        marker_ids   : int array of length n_channels, marker_ids[channel_id] = marker_id
        name2meta    : dict normalized_marker_name -> (mean, std, marker_id)
    """
    df = pd.read_csv(csv_path).sort_values("channel_id").reset_index(drop=True)
    n_ch = len(df)
    means      = df["marker_mean"].values.astype(np.float64)
    stds       = df["marker_std"].values.astype(np.float64)
    marker_ids = df["marker_id"].values.astype(np.int64)

    def _norm(s):
        return str(s).strip().upper().replace(" ", "").replace("-", "").replace("_", "")

    name2meta = {}
    for _, row in df.iterrows():
        key = _norm(row["marker_name"])
        name2meta[key] = (
            float(row["marker_mean"]),
            float(row["marker_std"]),
            int(row["marker_id"]),
        )

    print(f"Loaded {n_ch} markers from CSV  "
          f"(channel_id 0–{n_ch-1}, marker_id range {marker_ids.min()}–{marker_ids.max()})")
    return df, means, stds, marker_ids, name2meta


def _norm_marker_name(name):
    return str(name).strip().upper().replace(" ", "").replace("-", "").replace("_", "")


def _looks_like_dye_name(name):
    s = str(name).strip().upper()
    dye_tokens = ["FITC", "CY5", "ATTO", "ALEXA", "TRITC", "CY3"]
    return any(tok in s for tok in dye_tokens)


def resolve_marker_names_for_patient(patient_dir, n_ch):
    """Return channel-aligned marker names and the source used.

    Priority:
      1) channels.json with semantic marker names (best).
      2) Known panel fallback by channel count (25/55/56).
    """
    channels_json = patient_dir / "channels.json"
    if channels_json.exists():
        try:
            ch_names = json.loads(channels_json.read_text())
        except Exception:
            ch_names = None
        if isinstance(ch_names, list) and len(ch_names) >= n_ch:
            ch_names = ch_names[:n_ch]
            semantic_ratio = sum(not _looks_like_dye_name(x) for x in ch_names) / float(n_ch)
            if semantic_ratio >= 0.8:
                return ch_names, "channels.json"

    panel = PANEL_MARKERS_BY_CHANNEL_COUNT.get(n_ch)
    if panel is not None:
        return panel, f"panel_{n_ch}ch"

    raise ValueError(
        f"Cannot resolve semantic markers for {patient_dir.name} ({n_ch} channels): "
        "channels.json is non-semantic and no known panel mapping exists."
    )


def build_semantic_stats(marker_names, name2meta):
    """Map marker names -> per-channel (mean, std, marker_id) arrays."""
    means, stds, marker_ids = [], [], []
    missing = []
    for i, name in enumerate(marker_names):
        key = _norm_marker_name(name)
        if key not in name2meta:
            missing.append((i, name))
            continue
        m, s, mid = name2meta[key]
        means.append(m)
        stds.append(s)
        marker_ids.append(mid)

    if missing:
        preview = ", ".join([f"ch{i}:{n}" for i, n in missing[:10]])
        raise ValueError(
            f"Marker metadata missing {len(missing)} channels (examples: {preview}). "
            "Please update marker_metadata_csv or panel mapping."
        )

    means = np.array(means, dtype=np.float64)
    stds = np.array(stds, dtype=np.float64)
    marker_ids = np.array(marker_ids, dtype=np.int64)
    stds[stds == 0] = 1.0
    return means, stds, marker_ids


# ---------------------------------------------------------------------------
# Embedding inference
# ---------------------------------------------------------------------------

def extract_embeddings(model, patches, means, stds, marker_ids_per_channel, device, use_amp=True):
    """Run the model on a list of patches and return CLS-token embeddings.

    Each patch (C, H, W) is:
      1. Normalised to [0, 1] by dividing by the dtype maximum (uint8 → /255,
         uint16 → /65535, float → unchanged). This mirrors the preprocessing in
         data_augmentation.py used during training.
      2. Resized (interpolated) on H/W to the nearest upper multiple of 16.
      3. Z-score normalised per channel using the provided means/stds.
      4. Passed through the model with the semantic marker_ids from the training
         CSV (marker_ids_per_channel[c] = the marker_id that was used for
         channel c during training). This ensures each channel is looked up
         at the correct position in the model's marker embedding table.

    Returns an array of shape [N, D].
    """
    n_ch = patches[0].shape[0]
    means_t    = torch.tensor(means[:n_ch], dtype=torch.float32, device=device)
    stds_t     = torch.tensor(stds[:n_ch],  dtype=torch.float32, device=device)
    marker_ids = marker_ids_per_channel[:n_ch].tolist()
    all_emb = []

    # Scale factor: normalize integer patches to [0, 1] before Z-score,
    # matching the training pipeline in data_augmentation.py (Normalization class)
    # which divides by np.iinfo(dtype).max for integer dtypes.
    # marker_metadata stats (means/stds) are in [0, 1] space.
    # float patches are assumed to already be in [0, 1] range (scale = 1.0).
    sample_dtype = patches[0].dtype
    if np.issubdtype(sample_dtype, np.integer):
        dtype_scale = float(np.iinfo(sample_dtype).max)
    else:
        dtype_scale = 1.0

    for patch in tqdm(patches, desc="    Embedding", leave=False):
        _, ph, pw = patch.shape
        base_h = max(((ph + 15) // 16) * 16, 16)
        base_w = max(((pw + 15) // 16) * 16, 16)

        # Keep patch on CPU until inside the try block to avoid OOM on tensor creation.
        patch_cpu = torch.tensor(patch.astype(np.float32) / dtype_scale,
                                 dtype=torch.float32).unsqueeze(0)  # CPU

        # Retry with progressively smaller spatial sizes when CUDA OOM occurs.
        # interpolate, normalise, and model forward are ALL inside try so every
        # step can trigger the next retry scale.
        retry_scales = [1.0, 0.85, 0.70, 0.60, 0.50, 0.40, 0.33, 0.25]
        emb = None
        last_oom = None
        for si, spatial_scale in enumerate(retry_scales):
            new_h = max((int(round(base_h * spatial_scale)) // 16) * 16, 16)
            new_w = max((int(round(base_w * spatial_scale)) // 16) * 16, 16)
            try:
                x = patch_cpu
                if ph != new_h or pw != new_w:
                    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
                x = x.to(device)
                x = (x - means_t[None, :, None, None]) / (stds_t[None, :, None, None] + 1e-8)
                with torch.no_grad():
                    if device.type == "cuda" and use_amp:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            emb, _, _ = model(x, marker_ids=[torch.tensor(marker_ids, device=device)])
                    else:
                        emb, _, _ = model(x, marker_ids=[torch.tensor(marker_ids, device=device)])
                if si > 0:
                    print(f"    OOM fallback: resized patch to {new_h}x{new_w}")
                break
            except torch.cuda.OutOfMemoryError as e:
                last_oom = e
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

        if emb is None:
            if last_oom is not None:
                raise last_oom
            raise RuntimeError("Failed to compute embedding for patch due to unknown error.")

        all_emb.append(emb.float().cpu().numpy())
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return np.concatenate(all_emb, axis=0)  # [N, D]


# ---------------------------------------------------------------------------
# Per-patient processing
# ---------------------------------------------------------------------------

def process_patient(patient_info, model, device, meta_means, meta_stds, meta_marker_ids,
                    min_islet_cells_strict=20, marker_name_to_meta=None, use_amp=True):
    """Load pre-saved patches for one patient, filter, embed, and return a DataFrame.

    Patch files are expected at:  patches_dir/<patient_id>/islet_<id>.npy

    If a metadata_csv is provided (ND patients), per-islet cell counts are used
    to drop pseudo-islets with fewer than min_islet_cells_strict cells.
    For patients without a metadata_csv all patches are kept.

    Returns a DataFrame with islet metadata columns followed by emb_0 … emb_D columns,
    or None if no patches are found.
    """
    t0  = time.time()
    pid = patient_info["patient_id"]
    label = patient_info.get("label", "unknown")
    patches_dir = Path(patient_info["patches_dir"]) / pid
    meta_csv    = patient_info.get("metadata_csv")

    print(f"\n  {pid}  ({label})")

    patch_files = sorted(patches_dir.glob("islet_*.npy"))
    if not patch_files:
        print(f"  SKIP — no patch files in {patches_dir}")
        return None

    patches = [np.load(f) for f in patch_files]

    # Build islet_id → metadata row lookup from CSV (ND patients only)
    meta_lookup = {}
    if meta_csv and Path(meta_csv).exists():
        df = pd.read_csv(meta_csv)
        for _, row in df[(df["patient_id"] == pid) & (df["label"] == 1)].iterrows():
            meta_lookup[int(row["islet_id"])] = row

    # Filter pseudo-islets and collect spatial metadata
    raw_patches, raw_meta = [], []
    for f, patch in zip(patch_files, patches):
        islet_id = int(f.stem.split("_")[1])
        row = meta_lookup.get(islet_id, {})
        n_cells = int(row.get("n_cells", 0))
        if n_cells > 0 and n_cells < min_islet_cells_strict:
            continue
        raw_patches.append(patch)
        raw_meta.append({
            "islet_id": islet_id,
            "y":        row.get("y",        0),
            "x":        row.get("x",        0),
            "patch_h":  patch.shape[1],
            "patch_w":  patch.shape[2],
            "center_x": row.get("center_x", 0),
            "center_y": row.get("center_y", 0),
            "radius":   row.get("radius",   0),
            "n_cells":  n_cells,
        })

    dropped = len(patch_files) - len(raw_patches)
    if dropped:
        print(f"  Filtered {dropped} pseudo-islets (n_cells < {min_islet_cells_strict})")

    patches, meta_list = raw_patches, raw_meta
    if not patches:
        print("  SKIP — no patches left after filtering")
        return None
    n_ch = patches[0].shape[0]

    # Semantic alignment: channel index -> marker name -> (mean, std, marker_id).
    # This avoids channel-order assumptions across panels.
    if marker_name_to_meta is not None:
        marker_names, source = resolve_marker_names_for_patient(patches_dir, n_ch)
        means, stds, marker_ids = build_semantic_stats(marker_names, marker_name_to_meta)
        print(f"  Marker alignment source: {source}")
        print(f"  First channels: {marker_names[:min(6, len(marker_names))]}")
    elif meta_means is not None and len(meta_means) >= n_ch:
        # Legacy fallback: direct channel-index truncation.
        means = meta_means[:n_ch]
        stds = meta_stds[:n_ch]
        marker_ids = meta_marker_ids[:n_ch]
        print("  WARNING: using legacy channel-index alignment (no semantic mapping)")
    else:
        # Last-resort fallback: local normalization + sequential marker IDs.
        all_px = np.concatenate([p.reshape(n_ch, -1) for p in patches], axis=1).astype(np.float32)
        means  = np.mean(all_px, axis=1)
        stds   = np.std(all_px,  axis=1)
        stds[stds == 0] = 1.0
        marker_ids = np.arange(4, 4 + n_ch, dtype=np.int64)
        print("  WARNING: no marker metadata; using local stats + sequential marker_ids")

    embeddings = extract_embeddings(model, patches, means, stds, marker_ids, device, use_amp=use_amp)

    df_out = pd.DataFrame(meta_list)
    df_out["patient_id"] = pid
    df_out["label"]      = label
    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])

    torch.cuda.empty_cache()
    print(f"  {len(patches)} islets  ({time.time()-t0:.1f}s)")
    return pd.concat([df_out, emb_df], axis=1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def derive_run_name(checkpoint_path):
    """Build a readable run name from the two parent dirs + stem of the checkpoint path."""
    p = Path(checkpoint_path)
    parts = [p.parent.parent.name, p.parent.name, p.stem]
    parts = [s for s in parts if s and s not in ("", ".", "/")]
    deduped = [parts[0]] if parts else ["checkpoint"]
    for seg in parts[1:]:
        if seg != deduped[-1]:
            deduped.append(seg)
    return "__".join(deduped)


def _patient_result_paths(patient_results_dir, patient_id):
    safe = patient_id.replace("/", "_")
    return (
        patient_results_dir / f"{safe}__metadata.csv",
        patient_results_dir / f"{safe}__embeddings.npy",
    )


def save_patient_result(df, patient_results_dir):
    """Persist one patient's result immediately for resumable extraction."""
    patient_id = str(df["patient_id"].iloc[0])
    meta_path, emb_path = _patient_result_paths(patient_results_dir, patient_id)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    meta_cols = [c for c in df.columns if not c.startswith("emb_")]
    df[meta_cols].to_csv(meta_path, index=False)
    np.save(emb_path, df[emb_cols].values.astype(np.float32))


def load_all_patient_results(patient_infos, patient_results_dir):
    """Load all per-patient saved results and concatenate to one DataFrame."""
    all_dfs = []
    for p in patient_infos:
        pid = p["patient_id"]
        meta_path, emb_path = _patient_result_paths(patient_results_dir, pid)
        if not (meta_path.exists() and emb_path.exists()):
            continue
        meta_df = pd.read_csv(meta_path)
        emb = np.load(emb_path)
        emb_df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
        all_dfs.append(pd.concat([meta_df, emb_df], axis=1))
    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run_name",   default=None)
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed precision on CUDA (default: enabled).")
    parser.add_argument("--start-from", default=None,
                        help="Start from this patient_id (inclusive), e.g. HPAP-095")
    parser.add_argument("--resume", action="store_true",
                        help="Skip patients that already have per-patient saved outputs.")
    args = parser.parse_args()

    script_dir  = Path(__file__).parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    checkpoint_path = args.checkpoint or config["checkpoint_path"]
    run_name   = args.run_name or derive_run_name(checkpoint_path)
    output_dir = Path(config.get("output_dir", script_dir / "output")) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("Islet Embedding Extraction")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Run name   : {run_name}")
    print(f"  Output     : {output_dir}")
    print("=" * 55)

    if (output_dir / "embeddings.npy").exists() and (output_dir / "metadata.csv").exists() \
            and (not args.resume) and (args.start_from is None):
        print(f"\n[SKIP] embeddings already exist in {output_dir}/")
        print("       Delete or rename that directory to re-run.")
        return

    patients = config.get("patients", [])
    missing = [p["patient_id"] for p in patients if "patches_dir" not in p]
    if missing:
        print(f"\n[ERROR] Missing 'patches_dir' for: {missing}")
        return

    if args.start_from:
        start_idx = None
        for i, p in enumerate(patients):
            if p.get("patient_id") == args.start_from:
                start_idx = i
                break
        if start_idx is None:
            raise ValueError(f"--start-from not found in config patients: {args.start_from}")
        patients = patients[start_idx:]

    model, device, _ = load_model(checkpoint_path, model_type=config.get("model_type", "vits16"))

    meta_means = meta_stds = meta_marker_ids = marker_name_to_meta = None
    if config.get("marker_metadata_csv"):
        _, meta_means, meta_stds, meta_marker_ids, marker_name_to_meta = \
            load_marker_metadata(config["marker_metadata_csv"])

    strict = config.get("min_islet_cells_strict", 20)
    use_amp = bool(config.get("use_amp", True)) and (not args.no_amp)
    patient_results_dir = output_dir / "patient_results"
    patient_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing {len(patients)} patients...")

    for idx, pinfo in enumerate(patients, 1):
        print(f"[{idx}/{len(patients)}]", end="")
        pid = pinfo["patient_id"]
        if args.resume:
            meta_path, emb_path = _patient_result_paths(patient_results_dir, pid)
            if meta_path.exists() and emb_path.exists():
                print(f"\n  {pid}  (resume skip: already saved)")
                continue
        df = process_patient(pinfo, model, device, meta_means, meta_stds, meta_marker_ids,
                             min_islet_cells_strict=strict,
                             marker_name_to_meta=marker_name_to_meta,
                             use_amp=use_amp)
        if df is not None:
            save_patient_result(df, patient_results_dir)
            print(f"  Saved incremental result for {pid}")

    combined = load_all_patient_results(config.get("patients", []), patient_results_dir)
    if combined is None:
        print("\nNo data extracted — check paths in config.yaml")
        return

    emb_cols   = [c for c in combined.columns if c.startswith("emb_")]
    meta_cols  = [c for c in combined.columns if not c.startswith("emb_")]
    embeddings = combined[emb_cols].values

    combined[meta_cols].to_csv(output_dir / "metadata.csv", index=False)
    np.save(output_dir / "embeddings.npy", embeddings)

    # Also save disease-specific subsets for downstream analyses.
    for disease in ["ND", "AAB", "T1D", "T2D"]:
        sub = combined[combined["label"] == disease].copy()
        if sub.empty:
            continue
        sub_emb_cols = [c for c in sub.columns if c.startswith("emb_")]
        sub_meta_cols = [c for c in sub.columns if not c.startswith("emb_")]
        np.save(output_dir / f"embeddings_{disease}.npy", sub[sub_emb_cols].values)
        sub[sub_meta_cols].to_csv(output_dir / f"metadata_{disease}.csv", index=False)

    print(f"\n{'='*55}")
    print(f"Total islet patches : {len(combined)}")
    for label in ["ND", "AAB", "T1D", "T2D"]:
        n = (combined["label"] == label).sum()
        if n:
            print(f"  {label}: {n}")
    print(f"Embeddings shape    : {embeddings.shape}")
    print(f"Saved → {output_dir}/")


if __name__ == "__main__":
    main()
