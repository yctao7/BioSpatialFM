#!/bin/bash
#SBATCH --job-name=kronos_best
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=4
#SBATCH --mem=512g
#SBATCH --time=20:00:00
#SBATCH --account=drjieliu_owned1
#SBATCH --partition=drjieliu-h200
#SBATCH --mail-type=NONE
#SBATCH --output=/nfs/turbo/umms-drjieliu1/usr/peterszj/BioSpatialFM/finetune_best.log

cd /nfs/turbo/umms-drjieliu1/usr/peterszj/BioSpatialFM
source /nfs/turbo/umms-drjieliu1/usr/peterszj/BioSpatialFM/.venv/bin/activate

# ---- Data paths ----
DATA_PATH="/nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/CODEX/patches"
DATA_PATH_IMC="/nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/IMC/patches_05um"
DATA_PATH_ISLET="/nfs/turbo/umms-drjieliu1/usr/peterszj/BioSpatialFM/islet_patches_h5"

# Marker metadata
MARKER_METADATA="/nfs/turbo/umms-drjieliu1/usr/yctao/BioSpatialFM/tutorials/codex_dataset/dataset/marker_info_with_metadata.csv"
MARKER_METADATA_IMC="/nfs/turbo/umms-drjieliu1/usr/yctao/BioSpatialFM/tutorials/imc_dataset/dataset/marker_info_with_metadata.csv"

# ---- Sampling fractions (per-GPU batch = 48) ----
# primary_per_batch   = 48 - round(48*0.15) - round(48*0.06) = 48 - 7 - 3 = 38 CODEX
# secondary_per_batch = 7  IMC    (CODEX:IMC = 38:7 ≈ 5.4:1,  ~2.1x cycles/epoch)
# tertiary_per_batch  = 3  islet  (CODEX:islet = 38:3 ≈ 12.7:1, ~3.6x cycles/epoch)
IMC_FRACTION=0.15
ISLET_FRACTION=0.06

# ---- Output ----
OUTPUT_DIR="/nfs/turbo/umms-drjieliu1/usr/peterszj/BioSpatialFM/output/best"
RESUME_PATH="/nfs/turbo/umms-drjieliu1/usr/peterszj/BioSpatialFM/output/best/checkpoint_latest.pth"

MODEL_TYPE="vits16"
PRETRAINED_WEIGHTS="/nfs/turbo/umms-drjieliu1/usr/peterszj/BioSpatialFM/model_assets/models--MahmoodLab--kronos/snapshots/8edc2719ad67b2e2b766073b35c6cf8e6f5da516/kronos_vits16_model.pt"

# ---- WandB ----
WANDB_PROJECT="KRONOS-Finetune"
WANDB_RUN_NAME="best_3way"

# ---- Training (4× H200) ----
BATCH_SIZE=48
EPOCHS=10
LR=0.0005
MIN_LR=1e-6
NUM_WORKERS=10
WARMUP_EPOCHS=0

# ---- Regularization ----
WEIGHT_DECAY=0.04
WEIGHT_DECAY_END=0.1

# ---- DINO / MIM ----
MIM_LOSS_WEIGHT=1.0
MASK_RATIO=0.4
FREEZE_LAST_LAYER=0

# ---- KoLeo ----
KOLEO_LOSS_WEIGHT=0.1

# ---- Channel-mask consistency loss (Lmask) ----
MASK_LOSS_WEIGHT=1.0
MASK_WARMUP_EPOCHS=0
MASK_KEEP_MIN=0.05
MASK_KEEP_MAX=0.4
MASK_MIN_KEPT_CHANNELS=3
MASK_N_STUDENT_VIEWS=8
MASK_ALWAYS_KEEP_MARKERS="DAPI,DNA"

# ---- Misc ----
GRAD_ACCUM_STEPS=1
WARMUP_TEACHER_TEMP=0.04
TEACHER_TEMP=0.04
WARMUP_TEACHER_TEMP_EPOCHS=1

torchrun --nproc_per_node=4 --master-addr=127.0.0.1 --master-port=29600 \
    /nfs/turbo/umms-drjieliu1/usr/peterszj/BioSpatialFM/finetune_kronos_best.py \
    --distributed \
    --bf16 \
    --data_path "${DATA_PATH}" \
    --data_path_imc "${DATA_PATH_IMC}" \
    --data_path_islet "${DATA_PATH_ISLET}" \
    --marker_metadata "${MARKER_METADATA}" \
    --marker_metadata_imc "${MARKER_METADATA_IMC}" \
    --imc_fraction ${IMC_FRACTION} \
    --islet_fraction ${ISLET_FRACTION} \
    --output_dir "${OUTPUT_DIR}" \
    --model_type ${MODEL_TYPE} \
    --pretrained_weights ${PRETRAINED_WEIGHTS} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --min_lr ${MIN_LR} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --weight_decay ${WEIGHT_DECAY} \
    --weight_decay_end ${WEIGHT_DECAY_END} \
    --freeze_last_layer ${FREEZE_LAST_LAYER} \
    --num_workers ${NUM_WORKERS} \
    --mim_loss_weight ${MIM_LOSS_WEIGHT} \
    --mask_ratio ${MASK_RATIO} \
    --mask_global_crops_only \
    --koleo_loss_weight ${KOLEO_LOSS_WEIGHT} \
    --mask_loss_weight ${MASK_LOSS_WEIGHT} \
    --mask_warmup_epochs ${MASK_WARMUP_EPOCHS} \
    --mask_keep_min ${MASK_KEEP_MIN} \
    --mask_keep_max ${MASK_KEEP_MAX} \
    --mask_min_kept_channels ${MASK_MIN_KEPT_CHANNELS} \
    --mask_n_student_views ${MASK_N_STUDENT_VIEWS} \
    --mask_always_keep_markers "${MASK_ALWAYS_KEEP_MARKERS}" \
    --grad_accum_steps ${GRAD_ACCUM_STEPS} \
    --warmup_teacher_temp ${WARMUP_TEACHER_TEMP} \
    --teacher_temp ${TEACHER_TEMP} \
    --warmup_teacher_temp_epochs ${WARMUP_TEACHER_TEMP_EPOCHS} \
    --saveckp_freq 1 \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name ${WANDB_RUN_NAME} \
    ${RESUME_PATH:+--resume ${RESUME_PATH}}
