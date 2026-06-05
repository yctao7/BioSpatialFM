"""
Recompute marker stats over the full HPAP CODEX dataset using the fixed
compute_stats (now normalizes uint8 & uint16 to [0,1] consistently), then
re-derive marker_info_with_metadata.csv via the same mapping logic as
tutorials/1 - Data-Download-And-Preprocessing_codex.ipynb.

Run headless from SLURM; no GPU needed.
"""
import os
import sys
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
TUTORIALS_DIR = HERE  # script now lives inside tutorials/
sys.path.insert(0, TUTORIALS_DIR)
os.chdir(TUTORIALS_DIR)

from utils import MarkerMetadata
from utils.codex_dataset_prep import compute_stats


HPAP_INPUT = "/nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/CODEX/hpapdata"
PROJECT_DIR = "./codex_dataset"
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
LOCAL_METADATA_SRC = "../model_assets/marker_metadata.csv"


def step1_compute_stats():
    print("=" * 80)
    print("STEP 1: compute_stats over HPAP CODEX")
    print("=" * 80, flush=True)
    stats = compute_stats(HPAP_INPUT, DATASET_DIR)
    shutil.copy(LOCAL_METADATA_SRC, os.path.join(DATASET_DIR, "marker_metadata.csv"))
    print("Wrote marker_info.csv and copied marker_metadata.csv", flush=True)
    return stats


def step2_match_and_map(stats):
    print("=" * 80)
    print("STEP 2: match markers, apply mapping & manual IDs, export")
    print("=" * 80, flush=True)

    marker_info_csv = f"{PROJECT_DIR}/dataset/marker_info.csv"
    marker_metadata_csv = f"{PROJECT_DIR}/dataset/marker_metadata.csv"

    obj = MarkerMetadata(marker_info_csv, marker_metadata_csv, top_suggestions=5)
    obj.get_marker_metadata()
    print(f"Initial unmatched markers: {len(obj.missing_marker_dict)}", flush=True)

    obj.missing_marker_dict |= {
        "ACTA2": "A-SMA",
        "ECAD": "E-CADHERIN",
        "KRT": "CYTOKERATIN",
        "LAM": "LAMINA",
        "HLA-DR": "HLA_DR",
        "PD-L1": "PDL1",
        "VIM": "VIMENTIN",
        "MMR": "CD206",
    }
    obj.get_marker_metadata_with_mapping()
    print(f"After name mapping, still unmatched: {len(obj.missing_marker_dict)}", flush=True)

    new_marker_id_map = {
        "ARX": 9, "CDX2": 15, "ISL1": 43, "IRX2": 45, "NEUROD1": 49,
        "NKX2-2": 51, "NKX6-1": 53, "PAX6": 55, "PDX1": 57, "SOX9": 59,
        "ATP1A1": 141, "TUBB3": 145, "CD117": 177, "CD141": 191, "DPP4": 225,
        "MCAM": 237, "CD90": 239, "CD39L3": 247, "SELP": 277, "COL1A1": 311,
        "COL4A1": 313, "COL6": 315, "HSPG2": 317, "CHGA": 321, "GCG": 323,
        "GHRL": 325, "GP2": 327, "NPY": 329, "PNLIP": 331, "PPY": 333,
        "SST": 335, "SYP": 337, "PROINS": 339, "CPEP": 341,
    }

    marker_id_map = {
        marker: marker_id
        for marker, marker_id in zip(obj.marker_info["marker_name"], obj.marker_info["marker_id"])
        if marker_id != 0
    } | new_marker_id_map

    marker_metadata_dict = {
        marker: {
            "marker_id": marker_id,
            "marker_mean": stats.loc[stats["marker_name"] == marker, "marker_mean"].item(),
            "marker_std": stats.loc[stats["marker_name"] == marker, "marker_std"].item(),
        }
        for marker, marker_id in marker_id_map.items()
    }

    obj.set_marker_metadata(marker_metadata_dict)
    if len(obj.missing_marker_dict) > 0:
        print(f"Still unmatched after set_marker_metadata: {len(obj.missing_marker_dict)}", flush=True)
        print(obj.missing_marker_dict, flush=True)
    else:
        print("All markers mapped successfully.", flush=True)

    out_csv = f"{PROJECT_DIR}/dataset/marker_info_with_metadata.csv"
    obj.export_marker_metadata(out_csv)
    print(f"Exported: {out_csv}", flush=True)
    print(obj.marker_info, flush=True)


if __name__ == "__main__":
    stats = step1_compute_stats()
    step2_match_and_map(stats)
    print("\nDONE", flush=True)
