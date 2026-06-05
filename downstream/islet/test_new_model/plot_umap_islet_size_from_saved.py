"""
Plot islet-size distribution on a fixed UMAP layout.

Default workflow for shape consistency:
1) Run plot_umap_from_saved.py once to create fixed artifacts under
   output/umap_<run_name>/umap2d_*.npy
2) Run this script to overlay islet size on the same coordinates.
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


def parse_islet_id(name):
    m = re.search(r"(\d+)", str(name))
    return int(m.group(1)) if m else None


def pick_size_column(df):
    cols = list(df.columns)
    for c in cols:
        if "area" in c.lower():
            return c
    for c in cols:
        if c.lower() == "iletsize":
            return c
    for c in cols:
        if "size" in c.lower() or "diameter" in c.lower():
            return c
    raise ValueError("No size-like column found in islets.csv")


def build_size_table(islets_csv):
    raw = pd.read_csv(islets_csv)
    size_col = pick_size_column(raw)

    tbl = pd.DataFrame({
        "patient_id": raw["Image"].map(parse_patient_id),
        "islet_id": raw["Name"].map(parse_islet_id),
        "islet_size": pd.to_numeric(raw[size_col], errors="coerce"),
    }).dropna(subset=["patient_id", "islet_id", "islet_size"])

    tbl = tbl.groupby(["patient_id", "islet_id"], as_index=False)["islet_size"].median()
    return tbl, size_col


def load_fixed_umap(coords_dir):
    coords_dir = Path(coords_dir)
    coords = np.load(coords_dir / "umap2d_coords.npy")
    labels = np.load(coords_dir / "umap2d_labels.npy", allow_pickle=True).astype(str)
    patients = np.load(coords_dir / "umap2d_patient_ids.npy", allow_pickle=True).astype(str)

    meta_path = coords_dir / "umap2d_metadata_filtered.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Re-run plot_umap_from_saved.py to save filtered metadata."
        )
    meta = pd.read_csv(meta_path)
    return coords, labels, patients, meta


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
    return coords, labels, patients, meta


def main():
    here = Path(__file__).parent
    default_run = "finetune3__codex_combined__checkpoint_latest"

    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", default=str(here / "output" / default_run))
    parser.add_argument("--coords_dir", default=str(here / "output" / f"umap_{default_run}"))
    parser.add_argument("--islets_csv", default="/nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/CODEX/export_metadata/islets.csv")
    parser.add_argument("--output_dir", default=str(here / "output" / f"umap_islet_size_{default_run}"))
    parser.add_argument("--scope", choices=["nd", "aab", "t1d", "t2d", "all", "all_diseases"],
                        default="all_diseases",
                        help="Subset to plot: nd, aab, t1d, t2d, all, or all_diseases (2x2 grid). Default: all_diseases")
    parser.add_argument("--clip_top_pct", type=float, default=0.0,
                        help="Clip top x%% values for coloring (e.g., 1.0 means top 1%% clipped).")
    parser.add_argument("--n_neighbors", type=int, default=30)
    parser.add_argument("--min_dist", type=float, default=0.3)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--fit_if_missing", action="store_true",
                        help="If fixed UMAP artifacts are missing, fit a new UMAP as fallback.")
    args = parser.parse_args()

    coords_dir = Path(args.coords_dir)
    if (coords_dir / "umap2d_coords.npy").exists() and (coords_dir / "umap2d_labels.npy").exists() \
            and (coords_dir / "umap2d_patient_ids.npy").exists() and (coords_dir / "umap2d_metadata_filtered.csv").exists():
        coords, labels, patients, meta = load_fixed_umap(coords_dir)
        print(f"Using fixed UMAP layout from: {coords_dir}")
    else:
        if not args.fit_if_missing:
            raise FileNotFoundError(
                f"Fixed UMAP artifacts not complete in {coords_dir}. "
                "Run plot_umap_from_saved.py first (or pass --fit_if_missing)."
            )
        print("Fixed UMAP artifacts not found; fitting new UMAP (shape may differ).")
        coords, labels, patients, meta = fallback_fit_umap(
            args.emb_dir, args.n_neighbors, args.min_dist, args.metric
        )

    if "islet_id" not in meta.columns:
        raise ValueError("metadata must contain 'islet_id' to match islet size")

    size_tbl, size_col = build_size_table(args.islets_csv)

    work = meta.copy()
    work["islet_id"] = pd.to_numeric(work["islet_id"], errors="coerce")
    work["label"] = labels

    merged = work.merge(size_tbl, on=["patient_id", "islet_id"], how="left")
    sizes = pd.to_numeric(merged["islet_size"], errors="coerce").values

    # ------------------------------------------------------------------ #
    # all_diseases: 2x2 grid, each disease in its own color gradient      #
    # ------------------------------------------------------------------ #
    if args.scope == "all_diseases":
        disease_config = [
            ("nd",  "ND"),
            ("aab", "AAB"),
            ("t1d", "T1D"),
            ("t2d", "T2D"),
        ]
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())

        def _disease_vals(dlabel):
            """Return (mask, clipped_values) for one disease type."""
            d_mask = (labels == dlabel) & np.isfinite(sizes)
            d_vals = sizes.copy()
            if args.clip_top_pct and args.clip_top_pct > 0 and d_mask.sum() > 0:
                vmax = np.nanpercentile(d_vals[d_mask], 100.0 - float(args.clip_top_pct))
                d_vals = np.minimum(d_vals, vmax)
            return d_mask, d_vals

        # --- Scatter 2×2 ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        for ax, (dkey, dlabel) in zip(axes.flatten(), disease_config):
            d_mask, d_vals = _disease_vals(dlabel)
            ax.scatter(coords[~d_mask, 0], coords[~d_mask, 1],
                       c="#D3D3D3", s=4, alpha=0.2, edgecolors="none")
            if d_mask.sum() > 0:
                sc = ax.scatter(
                    coords[d_mask, 0], coords[d_mask, 1],
                    c=d_vals[d_mask], cmap=_SCOPE_CMAP[dkey],
                    s=10, alpha=0.85, edgecolors="none",
                )
                cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(f"Islet size ({size_col})")
                print(f"  {dlabel}: {d_mask.sum()} points with size")
            else:
                print(f"  {dlabel}: no points with size — skipped")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f"{dlabel} — Islet Size")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
        fig.suptitle(f"Fixed UMAP: Islet Size by Disease Type ({size_col})", fontsize=14)
        plt.tight_layout()
        fig.savefig(out / "fixed_umap_all_diseases_islet_size_scatter.png", dpi=220)
        plt.close(fig)

        # --- Hexbin 2×2 ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        for ax, (dkey, dlabel) in zip(axes.flatten(), disease_config):
            d_mask, d_vals = _disease_vals(dlabel)
            ax.scatter(coords[~d_mask, 0], coords[~d_mask, 1],
                       c="#E0E0E0", s=4, alpha=0.15, edgecolors="none")
            if d_mask.sum() > 0:
                hb = ax.hexbin(
                    coords[d_mask, 0], coords[d_mask, 1],
                    C=d_vals[d_mask],
                    gridsize=55,
                    reduce_C_function=np.nanmean,
                    cmap=_SCOPE_CMAP[dkey],
                    mincnt=1,
                )
                cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label(f"Mean islet size ({size_col})")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f"{dlabel} — Islet Size Hexbin (mean per bin)")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
        fig.suptitle(f"Fixed UMAP: Islet Size Hexbin by Disease Type ({size_col})", fontsize=14)
        plt.tight_layout()
        fig.savefig(out / "fixed_umap_all_diseases_islet_size_hexbin.png", dpi=220)
        plt.close(fig)

        print(f"Saved to: {out}")
        return

    # ------------------------------------------------------------------ #
    # Single-scope mode (nd / aab / t1d / t2d / all)                      #
    # ------------------------------------------------------------------ #
    _scope_map = {
        "nd":  ("ND",  labels == "ND"),
        "aab": ("AAB", labels == "AAB"),
        "t1d": ("T1D", labels == "T1D"),
        "t2d": ("T2D", labels == "T2D"),
        "all": ("All", np.ones(len(labels), dtype=bool)),
    }
    scope_name, scope_mask = _scope_map[args.scope]

    size_mask = scope_mask & np.isfinite(sizes)
    n_scope = int(scope_mask.sum())
    n_scope_size = int(size_mask.sum())
    print(f"Using size column: {size_col}")
    print(f"{scope_name} points: {n_scope}; {scope_name} points with size: {n_scope_size}")

    if n_scope_size == 0:
        raise ValueError(f"No {scope_name} points with valid size found.")

    # Optional upper-tail clipping for better color contrast.
    color_values = sizes.copy()
    if args.clip_top_pct and args.clip_top_pct > 0:
        pct = float(args.clip_top_pct)
        if pct >= 100:
            raise ValueError("--clip_top_pct must be < 100")
        vmax = np.nanpercentile(color_values[size_mask], 100.0 - pct)
        color_values = np.minimum(color_values, vmax)
        print(f"Color clipping enabled: top {pct:.2f}% clipped at {vmax:.3f}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())

    bg = ~size_mask

    # Scatter on fixed layout
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(coords[bg, 0], coords[bg, 1], c="#D3D3D3", s=8, alpha=0.25, edgecolors="none")
    sc = ax.scatter(
        coords[size_mask, 0],
        coords[size_mask, 1],
        c=color_values[size_mask],
        cmap=_SCOPE_CMAP[args.scope],
        s=12,
        alpha=0.85,
        edgecolors="none",
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Fixed UMAP: {scope_name} patches colored by Islet Size")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(f"Islet size ({size_col})")
    plt.tight_layout()
    fig.savefig(out / f"fixed_umap_{args.scope}_islet_size_scatter.png", dpi=220)
    plt.close(fig)

    # Hexbin on fixed layout
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(coords[bg, 0], coords[bg, 1], c="#E0E0E0", s=6, alpha=0.15, edgecolors="none")
    hb = ax.hexbin(
        coords[size_mask, 0],
        coords[size_mask, 1],
        C=color_values[size_mask],
        gridsize=55,
        reduce_C_function=np.nanmean,
        cmap=_SCOPE_CMAP[args.scope],
        mincnt=1,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Fixed UMAP: {scope_name} islet-size hexbin (mean per bin)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label(f"Mean islet size ({size_col})")
    plt.tight_layout()
    fig.savefig(out / f"fixed_umap_{args.scope}_islet_size_hexbin.png", dpi=220)
    plt.close(fig)

    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
