#!/bin/bash
#SBATCH --job-name=extract_best
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH --time=2:00:00
#SBATCH --account=drjieliu_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --mail-type=NONE
#SBATCH --output=/nfs/turbo/umms-drjieliu1/usr/peterszj/BioSpatialFM/extract_best.log

source /nfs/turbo/umms-drjieliu1/usr/peterszj/BioSpatialFM/.venv/bin/activate

cd /nfs/turbo/umms-drjieliu1/usr/peterszj/BioSpatialFM/downstream/islet/test_new_model

echo "========================================"
echo "Extract embeddings: best/checkpoint_latest"
echo "Checkpoint: /nfs/turbo/umms-drjieliu1/usr/peterszj/BioSpatialFM/output/best/checkpoint_latest.pth"
echo "========================================"

python extract_embeddings.py \
    --config config_best.yaml \
    --resume

echo ""
echo "Done. Results saved to:"
echo "  /nfs/turbo/umms-drjieliu1/usr/peterszj/BioSpatialFM/downstream/islet/test_new_model/output/best__checkpoint_latest/"
