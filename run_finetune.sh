#!/bin/bash
<<<<<<< HEAD
#SBATCH --job-name=kronos_finetune
=======
#SBATCH --job-name JOBNAME
>>>>>>> upstream/mim-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=256g
#SBATCH --time=12:00:00
#SBATCH --account=drjieliu_owned1
#SBATCH --partition=drjieliu-h200
#SBATCH --mail-type=NONE
<<<<<<< HEAD
#SBATCH --output=finetune.log
=======
#SBATCH --output=finetune3.log
>>>>>>> upstream/mim-dev

# Example script for fine-tuning KRONOS model
# Modify the parameters according to your needs

<<<<<<< HEAD
# Basic settings (CODEX only, H200)
DATA_PATH="/scratch/drjieliu_owned_root/drjieliu_owned1/yctao/patches/"
DATA_PATH_IMC="/scratch/drjieliu_owned_root/drjieliu_owned1/peterszj/islet_patches_h5"
MARKER_METADATA="tutorials/codex_dataset/dataset/marker_info_with_metadata.csv"
OUTPUT_DIR="output/finetune3/codex_combined"
=======
# Basic settings
DATA_PATH="/scratch/drjieliu_owned_root/drjieliu_owned1/yctao/patches"  # Change this to your data path
OUTPUT_DIR="output/finetune_0202_128"
>>>>>>> upstream/mim-dev
MODEL_TYPE="vits16"  # or "vitl16"
PRETRAINED_WEIGHTS="./model_assets/models--MahmoodLab--kronos/snapshots/8edc2719ad67b2e2b766073b35c6cf8e6f5da516/kronos_vits16_model.pt"
RESUME_PATH=""  

<<<<<<< HEAD
# WandB settings
# NOTE: 必须设置 WANDB_PROJECT 才会同步到 wandb（不能为空）
# 运行前确认已登录：wandb login
# metrics 会出现在 wandb Charts 下：iter_loss, iter_cls_loss, iter_mim_loss, iter_lr, iter_wd, epoch_loss 等
WANDB_PROJECT="KRONOS-Finetune"
WANDB_RUN_NAME="codex_combined"

# Training settings (2x H200: batch_size per GPU=32, effective batch=64)
#todo: change this part
BATCH_SIZE=32
EPOCHS=20
LR=0.0005            
MIN_LR=1e-6         # cosine decay 终点
NUM_WORKERS=10
WARMUP_EPOCHS=1     # ~10% of epochs

# Regularization
#todo: change this part
WEIGHT_DECAY=0.04
WEIGHT_DECAY_END=0.1   # fine-tuning 后期不需要强正则

=======
# Training settings
BATCH_SIZE=64
EPOCHS=100
LR=0.0005
>>>>>>> upstream/mim-dev

# DINO settings
MIM_LOSS_WEIGHT=1.0
<<<<<<< HEAD
MASK_RATIO=0.4
FREEZE_LAST_LAYER=0    # 去掉 freeze（ViT-S 不需要）

# KoLeo regularization
KOLEO_LOSS_WEIGHT=0.1

# Gradient accumulation
GRAD_ACCUM_STEPS=1

# Teacher temperature schedule (WARMUP_TEACHER_TEMP_EPOCHS 必须 <= EPOCHS)
#todo: change this part
WARMUP_TEACHER_TEMP=0.04
TEACHER_TEMP=0.04
WARMUP_TEACHER_TEMP_EPOCHS=1

# Run training (2x H200 distributed)
torchrun --nproc_per_node=2 --master-addr=127.0.0.1 --master-port=29500 finetune_kronos.py \
=======

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
>>>>>>> upstream/mim-dev
    --distributed \
    --data_path ${DATA_PATH} \
    --data_path_imc ${DATA_PATH_IMC} \
    --marker_metadata ${MARKER_METADATA} \
    --output_dir ${OUTPUT_DIR} \
    --model_type ${MODEL_TYPE} \
    --pretrained_weights ${PRETRAINED_WEIGHTS} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
<<<<<<< HEAD
    --min_lr ${MIN_LR} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --weight_decay ${WEIGHT_DECAY} \
    --weight_decay_end ${WEIGHT_DECAY_END} \
    --freeze_last_layer ${FREEZE_LAST_LAYER} \
    --num_workers ${NUM_WORKERS} \
    --out_dim ${OUT_DIM} \
    --local_crops_number ${LOCAL_CROPS_NUMBER} \
    --mim_loss_weight ${MIM_LOSS_WEIGHT} \
    --mask_ratio ${MASK_RATIO} \
    --mask_global_crops_only \
    --koleo_loss_weight ${KOLEO_LOSS_WEIGHT} \
    --grad_accum_steps ${GRAD_ACCUM_STEPS} \
    --warmup_teacher_temp ${WARMUP_TEACHER_TEMP} \
    --teacher_temp ${TEACHER_TEMP} \
    --warmup_teacher_temp_epochs ${WARMUP_TEACHER_TEMP_EPOCHS} \
    --saveckp_freq 4 \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME} \
    ${RESUME_PATH:+--resume ${RESUME_PATH}}

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
=======
    --mim_loss_weight ${MIM_LOSS_WEIGHT} \
>>>>>>> upstream/mim-dev
