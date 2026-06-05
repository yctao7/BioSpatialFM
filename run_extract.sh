#!/bin/bash
#SBATCH --job-name=imc_patch_extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=24:00:00
#SBATCH --account=drjieliu_owned1
#SBATCH --partition=standard
#SBATCH --mail-type=NONE
#SBATCH --output=extract_05um.log

# Extract IMC patches from raw multi-channel OME-TIFFs.
#   - hpapdata_comb has one .ome.tif per ROI with 34 channels in OME XML metadata.
#   - default --modality codex correctly handles this format (parses ome:Channel
#     names, writes uppercase canonical marker names as h5 dataset keys).
#   - smoke test (3 ROIs of HPAP-001): 34/34 marker names matched IMC CSV
#     (DNA, ACTB, KI67, NFKB, PS6, ...), ~0.9 s per ROI, ~16 patches/ROI.
#   - 671 ROIs total -> ~12 min compute + I/O margin -> 4h wall budget is generous.

cd /nfs/turbo/umms-drjieliu1/usr/yctao/BioSpatialFM
source .venv/bin/activate

python tutorials/utils/patch_extractor.py \
    --input_dir /nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/IMC/hpapdata_comb_05um \
    --output_dir /nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/IMC/patches_05um

# --- earlier extraction targets (kept commented for reference) ---
# CODEX raw -> patches:
# python tutorials/utils/patch_extractor.py \  # CODEX

#     --input_dir /nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/CODEX/hpapdata \
#     --output_dir /scratch/drjieliu_owned_root/drjieliu_owned1/peterszj/patches/codex
#
# IMC per-marker-tiff variant (NOT applicable to hpapdata_comb -- that path uses
# multi-channel ome.tif. --modality imc is for the per-marker-one-tiff layout):
# python tutorials/utils/patch_extractor.py \  # IMC per-marker
#     --input_dir /nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/IMC/hpapdata \
#     --output_dir /scratch/drjieliu_owned_root/drjieliu_owned1/peterszj/patches/imc \
#     --modality imc
