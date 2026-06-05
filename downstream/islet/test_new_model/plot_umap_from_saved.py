"""
Plot UMAP from saved embeddings.

Input (preferred):
  output/<run_name>/embeddings.npy + metadata.csv
Fallback (legacy):
  all_embeddings.npy + all_labels.npy + all_patient_ids.npy

Usage:
  python plot_umap_from_saved.py
  python plot_umap_from_saved.py --umap_3d
  python plot_umap_from_saved.py --investigate
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from umap import UMAP
from sklearn.preprocessing import normalize


DISEASE_COLORS = {"ND": "#4EAADB", "AAB": "#F5A623", "T1D": "#E8555A", "T2D": "#6BAF6B"}
DISEASE_ORDER  = ["ND", "AAB", "T1D", "T2D"]


# ─────────────────────────────────────────────────────────────
# 2D UMAP 标准图
# ─────────────────────────────────────────────────────────────

def plot_umap(coords, labels, patients, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1: 按病型
    fig, ax = plt.subplots(figsize=(9, 7))
    for disease in DISEASE_ORDER:
        mask = labels == disease
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=DISEASE_COLORS.get(disease, "#999"),
                   label=f"{disease} (n={mask.sum()})",
                   s=12, alpha=0.6, edgecolors="none")
    ax.set_title("Islet patches — by Disease Type", fontsize=13)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3, fontsize=10)
    plt.tight_layout()
    fig.savefig(output_dir / "umap_by_disease.png", dpi=200)
    plt.close(fig)
    print("  Saved: umap_by_disease.png")

    # 2: 每种病型单独高亮
    for spotlight in DISEASE_ORDER:
        if (labels == spotlight).sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(9, 7))
        bg_mask = labels != spotlight
        ax.scatter(coords[bg_mask, 0], coords[bg_mask, 1],
                   c="#CCCCCC", s=8, alpha=0.3, edgecolors="none", zorder=1)
        fg_mask = labels == spotlight
        ax.scatter(coords[fg_mask, 0], coords[fg_mask, 1],
                   c=DISEASE_COLORS[spotlight],
                   label=f"{spotlight} (n={fg_mask.sum()})",
                   s=14, alpha=0.8, edgecolors="none", zorder=2)
        ax.set_title(f"Islet patches — {spotlight} highlighted", fontsize=13)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.legend(markerscale=3, fontsize=11)
        plt.tight_layout()
        fig.savefig(output_dir / f"umap_spotlight_{spotlight}.png", dpi=200)
        plt.close(fig)
        print(f"  Saved: umap_spotlight_{spotlight}.png")

    print(f"  → {output_dir}/")


# ─────────────────────────────────────────────────────────────
# 3D UMAP → 2D 切片对比
# ─────────────────────────────────────────────────────────────

def plot_umap_3d_slices(coords3d, labels, output_dir):
    """从 3D UMAP 坐标生成三个两两投影切片，与直接 2D UMAP 对比。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = [(0, 1, "UMAP 1", "UMAP 2"),
             (0, 2, "UMAP 1", "UMAP 3"),
             (1, 2, "UMAP 2", "UMAP 3")]

    for i, j, xlabel, ylabel in pairs:
        fig, ax = plt.subplots(figsize=(9, 7))
        for disease in DISEASE_ORDER:
            mask = labels == disease
            if mask.sum() == 0:
                continue
            ax.scatter(coords3d[mask, i], coords3d[mask, j],
                       c=DISEASE_COLORS.get(disease, "#999"),
                       label=f"{disease} (n={mask.sum()})",
                       s=12, alpha=0.6, edgecolors="none")
        ax.set_title(f"3D UMAP slice — {xlabel} vs {ylabel}", fontsize=13)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.legend(markerscale=3, fontsize=10)
        plt.tight_layout()
        fname = f"umap3d_{xlabel.replace(' ', '')}_{ylabel.replace(' ', '')}.png"
        fig.savefig(output_dir / fname, dpi=200)
        plt.close(fig)
        print(f"  Saved: {fname}")

    print(f"  → {output_dir}/")


# ─────────────────────────────────────────────────────────────
# Cluster 诊断（HDBSCAN）
# ─────────────────────────────────────────────────────────────

def investigate_clusters(coords2d, labels, patients, patch_indices, output_dir,
                         min_cluster_size=30):
    """
    用 HDBSCAN 对 2D UMAP 坐标聚类，输出：
    - UMAP 按 cluster 编号着色（含 noise 点）
    - 每个 cluster 的病人/病型组成条形图
    - cluster 摘要 CSV（含 suspect batch effect 标记）
    """
    try:
        import hdbscan
    except ImportError:
        print("  [ERROR] hdbscan 未安装，请运行: pip install hdbscan")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Running HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                 min_samples=5,
                                 prediction_data=True)
    cluster_ids = clusterer.fit_predict(coords2d)

    unique_clusters = sorted(set(cluster_ids))
    n_clusters = sum(1 for c in unique_clusters if c >= 0)
    n_noise    = (cluster_ids == -1).sum()
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")

    # ── 1. UMAP 按 cluster 着色 ──
    cmap = cm.get_cmap("tab20", max(n_clusters, 1))
    fig, ax = plt.subplots(figsize=(11, 8))
    # noise 先画灰色
    noise_mask = cluster_ids == -1
    if noise_mask.sum() > 0:
        ax.scatter(coords2d[noise_mask, 0], coords2d[noise_mask, 1],
                   c="#CCCCCC", s=6, alpha=0.3, edgecolors="none",
                   label=f"noise (n={noise_mask.sum()})", zorder=1)
    for cid in unique_clusters:
        if cid < 0:
            continue
        mask = cluster_ids == cid
        ax.scatter(coords2d[mask, 0], coords2d[mask, 1],
                   color=cmap(cid), s=12, alpha=0.7, edgecolors="none",
                   label=f"C{cid} (n={mask.sum()})", zorder=2)
        # 在质心标注编号
        cx, cy = coords2d[mask, 0].mean(), coords2d[mask, 1].mean()
        ax.text(cx, cy, str(cid), fontsize=7, ha="center", va="center",
                fontweight="bold", color="black", zorder=3)
    ax.set_title(f"UMAP — HDBSCAN clusters ({n_clusters} clusters)", fontsize=13)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              markerscale=2, fontsize=6, ncol=2)
    plt.tight_layout()
    fig.savefig(output_dir / "umap_clusters.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: umap_clusters.png")

    # ── 2. 每个 cluster 的组成条形图 ──
    rows = []
    for cid in unique_clusters:
        mask = cluster_ids == cid
        tag  = "noise" if cid < 0 else f"C{cid}"
        pats_in = patients[mask]
        labs_in = labels[mask]
        idxs_in = patch_indices[mask]

        disease_counts = {d: int((labs_in == d).sum()) for d in DISEASE_ORDER}
        pat_counts     = {p: int((pats_in == p).sum()) for p in sorted(set(pats_in))}
        n_dominant_pat = max(pat_counts.values()) if pat_counts else 0
        pct_dominant   = 100 * n_dominant_pat / mask.sum() if mask.sum() > 0 else 0
        suspect        = (len(pat_counts) <= 2) and (pct_dominant >= 80)

        rows.append({
            "cluster":         tag,
            "n_patches":       int(mask.sum()),
            "n_patients":      len(pat_counts),
            **{f"n_{d}": disease_counts[d] for d in DISEASE_ORDER},
            "dominant_patient": max(pat_counts, key=pat_counts.get) if pat_counts else "",
            "dominant_pct":    round(pct_dominant, 1),
            "suspect_batch":   suspect,
            "patients":        "; ".join(f"{p}({c})" for p, c in sorted(pat_counts.items())),
            "patch_indices":   ",".join(map(str, sorted(idxs_in.tolist()))),
        })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(output_dir / "cluster_summary.csv", index=False)
    print("  Saved: cluster_summary.csv")

    # ── 3. 每个真实 cluster 的病型组成条形图 ──
    real_clusters = [r for r in rows if r["cluster"] != "noise"]
    if real_clusters:
        n_cols = min(4, len(real_clusters))
        n_rows = (len(real_clusters) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(3.5 * n_cols, 3 * n_rows))
        axes = np.array(axes).flatten()

        for i, row in enumerate(real_clusters):
            ax = axes[i]
            vals   = [row[f"n_{d}"] for d in DISEASE_ORDER]
            colors = [DISEASE_COLORS[d] for d in DISEASE_ORDER]
            bars   = ax.bar(DISEASE_ORDER, vals, color=colors)
            title  = f"{row['cluster']}  (n={row['n_patches']}, {row['n_patients']} pts)"
            if row["suspect_batch"]:
                title += "\n⚠ suspect batch"
            ax.set_title(title, fontsize=8)
            ax.set_ylabel("patches", fontsize=7)
            ax.tick_params(labelsize=7)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            str(val), ha="center", va="bottom", fontsize=6)

        # 关掉多余的子图
        for j in range(len(real_clusters), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Cluster Composition (disease type breakdown)", fontsize=11, y=1.01)
        plt.tight_layout()
        fig.savefig(output_dir / "cluster_composition.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: cluster_composition.png")

    # ── 4. 打印摘要 ──
    print(f"\n{'Cluster':<8} {'N':>6} {'Pts':>5} {'ND':>5} {'AAB':>5} {'T1D':>5} {'T2D':>5} {'Dominant patient':<20} {'Suspect'}")
    print("-" * 80)
    for row in rows:
        flag = "⚠" if row["suspect_batch"] else ""
        print(f"{row['cluster']:<8} {row['n_patches']:>6} {row['n_patients']:>5} "
              f"{row['n_ND']:>5} {row['n_AAB']:>5} {row['n_T1D']:>5} {row['n_T2D']:>5} "
              f"{row['dominant_patient']:<20} {flag}")

    print(f"\n  → {output_dir}/")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).parent
    default_run_name = "finetune3__codex_combined__checkpoint_latest"
    default_emb_dir = here / "output" / default_run_name
    default_output_dir = here / "output" / f"umap_{default_run_name}"

    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir",    default=str(default_emb_dir))
    parser.add_argument("--output_dir", default=str(default_output_dir))
    parser.add_argument("--n_neighbors",       type=int,   default=30)
    parser.add_argument("--min_dist",          type=float, default=0.3)
    parser.add_argument("--metric",            default="cosine")
    parser.add_argument("--umap_3d",           action="store_true",
                        help="同时拟合 3D UMAP 并生成三个切片图")
    parser.add_argument("--investigate",       action="store_true",
                        help="用 HDBSCAN 对 2D UMAP 聚类并输出诊断报告")
    parser.add_argument("--min_cluster_size",  type=int, default=30,
                        help="HDBSCAN min_cluster_size (default: 30)")
    parser.add_argument("--l2_norm",           action="store_true",
                        help="在 UMAP 前对 embeddings 做 L2 归一化")
    args = parser.parse_args()

    # 加载数据
    emb_dir = Path(args.emb_dir)
    print(f"Loading from: {emb_dir}")

    # New format from extract_embeddings.py
    emb_file = emb_dir / "embeddings.npy"
    meta_file = emb_dir / "metadata.csv"
    meta = None
    if emb_file.exists() and meta_file.exists():
        embeddings = np.load(emb_file)
        meta = pd.read_csv(meta_file)
        labels = meta["label"].astype(str).values
        patients = meta["patient_id"].astype(str).values
        print("  Source format: embeddings.npy + metadata.csv")
    else:
        # Legacy format fallback
        embeddings = np.load(emb_dir / "all_embeddings.npy")
        labels     = np.load(emb_dir / "all_labels.npy",      allow_pickle=True)
        patients   = np.load(emb_dir / "all_patient_ids.npy", allow_pickle=True)
        print("  Source format: all_embeddings.npy + all_labels.npy + all_patient_ids.npy")

    patch_indices = np.arange(len(embeddings))
    print(f"  embeddings: {embeddings.shape}")
    print(f"  labels:     {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Skip rows with NaN/Inf at runtime (do not modify source files).
    finite_mask = np.isfinite(embeddings).all(axis=1)
    n_bad = int((~finite_mask).sum())
    if n_bad > 0:
        print(f"  Non-finite rows detected: {n_bad} (NaN/Inf). They will be skipped.")
        embeddings = embeddings[finite_mask]
        labels = labels[finite_mask]
        patients = patients[finite_mask]
        patch_indices = patch_indices[finite_mask]
        if meta is not None:
            meta = meta.iloc[np.where(finite_mask)[0]].reset_index(drop=True)
        print(f"  embeddings after skip: {embeddings.shape}")

    if args.l2_norm:
        embeddings = normalize(embeddings, norm="l2")
        print("  L2 归一化完成")

    # 2D UMAP
    print(f"\nFitting 2D UMAP (n_neighbors={args.n_neighbors}, "
          f"min_dist={args.min_dist}, metric={args.metric}"
          f"{', l2_norm' if args.l2_norm else ''})...")
    reducer2d = UMAP(n_components=2, random_state=42,
                     n_neighbors=args.n_neighbors,
                     min_dist=args.min_dist,
                     metric=args.metric)
    coords2d = reducer2d.fit_transform(embeddings)

    suffix = "_l2norm" if args.l2_norm else ""
    out2d = Path(args.output_dir + suffix)
    print("\nGenerating 2D UMAP plots...")
    plot_umap(coords2d, labels, patients, out2d)

    # Save fixed 2D layout artifacts for reproducible downstream overlays.
    out2d.mkdir(parents=True, exist_ok=True)
    np.save(out2d / "umap2d_coords.npy", coords2d)
    np.save(out2d / "umap2d_labels.npy", labels.astype(str))
    np.save(out2d / "umap2d_patient_ids.npy", patients.astype(str))
    np.save(out2d / "umap2d_patch_indices.npy", patch_indices.astype(np.int64))
    if meta is not None:
        meta.to_csv(out2d / "umap2d_metadata_filtered.csv", index=False)
    print("  Saved: umap2d_coords.npy / umap2d_labels.npy / umap2d_patient_ids.npy")

    # 3D UMAP（可选）
    if args.umap_3d:
        print(f"\nFitting 3D UMAP ...")
        reducer3d = UMAP(n_components=3, random_state=42,
                         n_neighbors=args.n_neighbors,
                         min_dist=args.min_dist,
                         metric=args.metric)
        coords3d = reducer3d.fit_transform(embeddings)
        out3d = out2d / "3d_slices"
        print("Generating 3D UMAP slice plots...")
        plot_umap_3d_slices(coords3d, labels, out3d)

    # Cluster 诊断（可选）
    if args.investigate:
        inv_dir = here / "output/investigation"
        print(f"\nRunning cluster investigation → {inv_dir}/")
        investigate_clusters(coords2d, labels, patients, patch_indices,
                             inv_dir, min_cluster_size=args.min_cluster_size)


if __name__ == "__main__":
    main()
