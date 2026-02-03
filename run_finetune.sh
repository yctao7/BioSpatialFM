#!/bin/bash
#SBATCH --job-name JOBNAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=256g
#SBATCH --time=12:00:00
#SBATCH --account=drjieliu_owned1
#SBATCH --partition=drjieliu-h200
#SBATCH --mail-type=NONE
#SBATCH --output=finetune3.log

# Example script for fine-tuning KRONOS model
# Modify the parameters according to your needs

# Basic settings
DATA_PATH="/scratch/drjieliu_owned_root/drjieliu_owned1/yctao/patches"  # Change this to your data path
OUTPUT_DIR="output/finetune_0202_128"
MODEL_TYPE="vits16"  # or "vitl16"
PRETRAINED_WEIGHTS="./model_assets/models--MahmoodLab--kronos/snapshots/8edc2719ad67b2e2b766073b35c6cf8e6f5da516/kronos_vits16_model.pt"
#RESUME_PATH="output/finetune3/MIM_batch_adjusted/checkpoint_latest.pth"

# Training settings
BATCH_SIZE=64
EPOCHS=100
LR=0.0005

# DINO settings
MIM_LOSS_WEIGHT=1.0

# Run training
# python finetune_kronos.py \
#     --data_path ${DATA_PATH} \
#     --output_dir ${OUTPUT_DIR} \
#     --model_type ${MODEL_TYPE} \
#     --pretrained_weights ${PRETRAINED_WEIGHTS} \
#     --batch_size ${BATCH_SIZE} \
#     --epochs ${EPOCHS} \
#     --lr ${LR} \
#     --mim_loss_weight ${MIM_LOSS_WEIGHT} \

# For distributed training (multi-GPU), use:
torchrun --nproc_per_node=2 finetune_kronos.py \
    --distributed \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --model_type ${MODEL_TYPE} \
    --pretrained_weights ${PRETRAINED_WEIGHTS} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --mim_loss_weight ${MIM_LOSS_WEIGHT} \
