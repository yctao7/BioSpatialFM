"""
Plot age distribution on a fixed UMAP layout.

Default workflow for shape consistency:
1) Run plot_umap_from_saved.py once to create fixed artifacts under
   output/umap_<run_name>/umap2d_*.npy
2) Run this script to overlay age on the same coordinates.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from umap import UMAP

# Per-scope single-hue darkness colormaps (light=low, dark=high).
# Background points are gray, so each hue stays visually distinct at all brightness levels.
_SCOPE_CMAP = {
    "nd":  mcolors.LinearSegmentedColormap.from_list("nd_dark",  ["#c6dbef", "#08306b"]),  # blue
    "aab": mcolors.LinearSegmentedColormap.from_list("aab_dark", ["#fff7bc", "#8c5e00"]),  # yellow
    "t1d": mcolors.LinearSegmentedColormap.from_list("t1d_dark", ["#fcbba1", "#67000d"]),  # red
    "t2d": mcolors.LinearSegmentedColormap.from_list("t2d_dark", ["#c7e9c0", "#00441b"]),  # green
    "all": mcolors.LinearSegmentedColormap.from_list("all_dark", ["#c6dbef", "#08306b"]),  # blue
}


def parse_patient_id(text):
    m = re.search(r"HPAP[-_ ]?(\d{3})", str(text), flags=re.IGNORECASE)
    return f"HPAP-{m.group(1)}" if m else None


def parse_age_years_from_image(image_name):
    m = re.search(r"-(\d+)y", str(image_name), flags=re.IGNORECASE)
    return float(m.group(1)) if m else np.nan


def build_patient_age_map(islets_csv):
    df = pd.read_csv(islets_csv, usecols=["Image"]).dropna()
    df["patient_id"] = df["Image"].map(parse_patient_id)
    df["age_years"] = df["Image"].map(parse_age_years_from_image)
    df = df.dropna(subset=["patient_id", "age_years"])
    return df.groupby("patient_id", as_index=True)["age_years"].median().to_dict()


def load_fixed_umap(coords_dir):
    coords_dir = Path(coords_dir)
    coords = np.load(coords_dir / "umap2d_coords.npy")
    labels = np.load(coords_dir / "umap2d_labels.npy", allow_pickle=True).astype(str)
    patients = np.load(coords_dir / "umap2d_patient_ids.npy", allow_pickle=True).astype(str)
    return coords, labels, patients


def fallback_fit_umap(emb_dir, n_neighbors, min_dist, metric):
    emb_dir = Path(emb_dir)
    emb = np.load(emb_dir / "embeddings.npy")
    meta = pd.read_csv(emb_dir / "metadata.csv")

    finite = np.isfinite(emb).all(axis=1)
    emb = emb[finite]
    meta = meta.iloc[np.where(finite)[0]].reset_index(drop=True)

    labels = meta["label"].astype(str).values
    patients = meta["patient_id"].astype(str).values
    reducer = UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
    )
    coords = reducer.fit_transform(emb)
    return coords, labels, patients


def main():
    here = Path(__file__).parent
    default_run = "finetune3__codex_combined__checkpoint_latest"

    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", default=str(here / "output" / default_run))
    parser.add_argument("--coords_dir", default=str(here / "output" / f"umap_{default_run}"))
    parser.add_argument("--islets_csv", default="/nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/CODEX/export_metadata/islets.csv")
    parser.add_argument("--output_dir", default=str(here / "output" / f"umap_age_{default_run}"))
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--min_dist", type=float, default=0.3)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--fit_if_missing", action="store_true",
                        help="If fixed UMAP artifacts are missing, fit a new UMAP as fallback.")
    parser.add_argument("--scope", choices=["nd", "aab", "t1d", "t2d", "all"], default="all",
                        help="Subset to plot: nd, aab, t1d, t2d, or all. Default: all")
    args = parser.parse_args()

    coords_dir = Path(args.coords_dir)
    if (coords_dir / "umap2d_coords.npy").exists() and (coords_dir / "umap2d_labels.npy").exists() \
            and (coords_dir / "umap2d_patient_ids.npy").exists():
        coords, labels, patients = load_fixed_umap(coords_dir)
        print(f"Using fixed UMAP layout from: {coords_dir}")
    else:
        if not args.fit_if_missing:
            raise FileNotFoundError(
                f"Fixed UMAP artifacts not found in {coords_dir}. "
                "Run plot_umap_from_saved.py first (or pass --fit_if_missing)."
            )
        print("Fixed UMAP artifacts not found; fitting new UMAP (shape may differ).")
        coords, labels, patients = fallback_fit_umap(
            args.emb_dir, args.n_neighbors, args.min_dist, args.metric
        )

    age_map = build_patient_age_map(args.islets_csv)
    ages = np.array([age_map.get(pid, np.nan) for pid in patients], dtype=np.float64)

    # Scope selection with valid age
    _scope_map = {
        "nd":  ("ND",  labels == "ND"),
        "aab": ("AAB", labels == "AAB"),
        "t1d": ("T1D", labels == "T1D"),
        "t2d": ("T2D", labels == "T2D"),
        "all": ("All", np.ones(len(labels), dtype=bool)),
    }
    scope_name, scope_mask = _scope_map[args.scope]

    age_mask = scope_mask & np.isfinite(ages)
    n_scope = int(scope_mask.sum())
    n_scope_age = int(age_mask.sum())
    print(f"{scope_name} points: {n_scope}; {scope_name} points with age: {n_scope_age}")

    if n_scope_age == 0:
        raise ValueError(f"No {scope_name} points with valid age found.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Keep identical shape context: draw all points as gray background.
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())

    bg = ~age_mask

    # 1) Age scatter on fixed layout
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(coords[bg, 0], coords[bg, 1], c="#D3D3D3", s=8, alpha=0.25, edgecolors="none")
    sc = ax.scatter(
        coords[age_mask, 0],
        coords[age_mask, 1],
        c=ages[age_mask],
        cmap=_SCOPE_CMAP[args.scope],
        s=12,
        alpha=0.85,
        edgecolors="none",
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Fixed UMAP: {scope_name} patches colored by Age")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Age (years)")
    plt.tight_layout()
    fig.savefig(out / f"fixed_umap_{args.scope}_age_scatter.png", dpi=220)
    plt.close(fig)

    # 2) Age hexbin on fixed layout
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(coords[bg, 0], coords[bg, 1], c="#E0E0E0", s=6, alpha=0.15, edgecolors="none")
    hb = ax.hexbin(
        coords[age_mask, 0],
        coords[age_mask, 1],
        C=ages[age_mask],
        gridsize=55,
        reduce_C_function=np.nanmean,
        cmap=_SCOPE_CMAP[args.scope],
        mincnt=1,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Fixed UMAP: {scope_name} age hexbin (mean per bin)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Mean age (years)")
    plt.tight_layout()
    fig.savefig(out / f"fixed_umap_{args.scope}_age_hexbin.png", dpi=220)
    plt.close(fig)

    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
