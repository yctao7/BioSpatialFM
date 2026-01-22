5#!/bin/bash

# Example script for fine-tuning KRONOS model
# Modify the parameters according to your needs

# Basic settings
DATA_PATH="tutorials/codex_dataset/patches"  # Change this to your data path
OUTPUT_DIR="output/finetune3/MIM_batch_adjusted"
MODEL_TYPE="vits16"  # or "vitl16"
PRETRAINED_WEIGHTS="./model_assets/models--MahmoodLab--kronos/snapshots/8edc2719ad67b2e2b766073b35c6cf8e6f5da516/kronos_vits16_model.pt"
RESUME_PATH="output/finetune3/MIM_batch_adjusted/checkpoint_latest.pth"

# Training settings
BATCH_SIZE=4
EPOCHS=120
LR=0.001
NUM_WORKERS=10

# DINO settings
OUT_DIM=65536
LOCAL_CROPS_NUMBER=8

# Run training
python finetune_kronos.py \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --model_type ${MODEL_TYPE} \
    --pretrained_weights ${PRETRAINED_WEIGHTS} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --num_workers ${NUM_WORKERS} \
    --out_dim ${OUT_DIM} \
    --local_crops_number ${LOCAL_CROPS_NUMBER} \
    --resume ${RESUME_PATH} \
    --saveckp_freq 20

# For distributed training (multi-GPU), use:
# torchrun --nproc_per_node=4 finetune_kronos.py \
#     --distributed \
#     --data_path ${DATA_PATH} \
#     --output_dir ${OUTPUT_DIR} \
#     --model_type ${MODEL_TYPE} \
#     --batch_size ${BATCH_SIZE} \
#     --epochs ${EPOCHS} \
#     --lr ${LR} \
#     --num_workers ${NUM_WORKERS} \
#     --out_dim ${OUT_DIM} \
#     --local_crops_number ${LOCAL_CROPS_NUMBER} \
#     --saveckp_freq 20
