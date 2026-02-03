#!/bin/bash
#SBATCH --job-name JOBNAME
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=256g
#SBATCH --time=12:00:00
#SBATCH --account=drjieliu_owned1
#SBATCH --partition=drjieliu-h200
#SBATCH --mail-type=NONE
#SBATCH --output=extract.log

# mkdir tutorials/codex_dataset/multiplex_images/
# cp -r /nfs/turbo/umms-drjieliu/proj/HPAP-Spatial/CODEX/hpapdata/HPAP-007 tutorials/codex_dataset/multiplex_images/
# cp -r /nfs/turbo/umms-drjieliu/proj/HPAP-Spatial/CODEX/hpapdata/HPAP-009 tutorials/codex_dataset/multiplex_images/
# python tutorials/utils/patch_extractor.py --input_dir tutorials/codex_dataset/multiplex_images --output_dir tutorials/codex_dataset/patches

python tutorials/utils/patch_extractor.py --input_dir /nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/CODEX/hpapdata --output_dir /scratch/drjieliu_owned_root/drjieliu_owned1/yctao/patches