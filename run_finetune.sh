#!/bin/bash
#SBATCH --job-name=kronos_finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=512g
#SBATCH --time=96:00:00
#SBATCH --account=drjieliu_owned1
#SBATCH --partition=drjieliu-h200
#SBATCH --mail-type=NONE
#SBATCH --output=finetune_05um.log

# Example script for fine-tuning KRONOS model
# Modify the parameters according to your needs

# Activate the project venv (compute nodes start with a clean PATH that does
# not include the venv binaries — torchrun lives only in .venv/bin).
cd /nfs/turbo/umms-drjieliu1/usr/yctao/BioSpatialFM
source .venv/bin/activate

# Continue logging into the previous wandb run (498ytush) instead of creating a
# new one, so the loss curves connect across the OOM-restart boundary.
export WANDB_RESUME=allow
export WANDB_RUN_ID=498ytush

# Basic settings (CODEX only, H200)
# Primary data sources (concatenated). Different on-disk layouts:
#   - CODEX:  flat dir of *.h5 patches at the top level
#   - islet:  nested subdirs containing *.h5 patches
# finetune_kronos.py walks each path recursively, so both layouts work.
DATA_PATH=(
    "/nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/CODEX/patches"
    "/scratch/drjieliu_owned_root/drjieliu_owned1/peterszj/islet_patches_h5"
)
# Secondary (IMC) source. patches_05um is from the 2x bilinearly upsampled
# OME-TIFFs (1um -> 0.5um/pixel) extracted by run_extract.sh; flat dir of
# *.h5 patches with canonical marker names matching the IMC CSV.
DATA_PATH_IMC="/nfs/turbo/umms-drjieliu1/projects/HPAP-Spatial/IMC/patches_05um"
MARKER_METADATA="tutorials/codex_dataset/dataset/marker_info_with_metadata.csv"
# IMC has its own marker_id namespace (e.g. DNA=3 vs DAPI=4 in CODEX). Both CSVs
# get merged inside finetune_kronos.py so all marker names resolve correctly.
MARKER_METADATA_IMC="tutorials/imc_dataset/dataset/marker_info_with_metadata.csv"
OUTPUT_DIR="output/finetune4f/codex_combined_05um"
MODEL_TYPE="vits16"  # or "vitl16"
PRETRAINED_WEIGHTS="./model_assets/models--MahmoodLab--kronos/snapshots/8edc2719ad67b2e2b766073b35c6cf8e6f5da516/kronos_vits16_model.pt"
RESUME_PATH="output/finetune4f/codex_combined_05um/checkpoint_latest.pth"

# WandB settings
# NOTE: 必须设置 WANDB_PROJECT 才会同步到 wandb（不能为空）
# 运行前确认已登录：wandb login
# metrics 会出现在 wandb Charts 下：iter_loss, iter_cls_loss, iter_mim_loss, iter_lr, iter_wd, epoch_loss 等
WANDB_PROJECT="KRONOS-Finetune"
WANDB_RUN_NAME="codex_combined_05um"

# Training settings (2x H200: batch_size per GPU=48, effective batch=96)
BATCH_SIZE=48
# 10 epochs: previous run hit convergence elbow by epoch 1 and was effectively
# flat by epoch 7 (total loss 0.5 -> 0.4 over 6 more epochs). 10 leaves ~3-4
# epochs of refinement post-elbow without wasting compute on the zero-gain tail.
EPOCHS=10
LR=0.0005            # auto-scales with effective batch (linear scaling rule)
MIN_LR=1e-6
NUM_WORKERS=10
# No LR warmup: fine-tuning from pretrained KRONOS, not from-scratch. Previous
# run showed loss dropped 31 -> 5 inside the warmup window with LR ~6e-6,
# i.e. the model was learning before LR even came up; warmup just delayed
# the peak-LR refinement phase.
WARMUP_EPOCHS=0
# IMC fraction of each batch (rest is CODEX). At BATCH_SIZE=48:
#   IMC_FRACTION=0.17 -> round(48*0.17)=8 -> CODEX:IMC = 5:1  (40 + 8)
# Equivalent to 1/6 after rounding. Plain DistributedSampler would give ~2% IMC
# per batch given the 690k:12k pool size mismatch -- way too sparse for mask CE
# to learn IMC panel invariance. Other useful values at batch=48:
#   0.50 -> 1:1 (24+24), 0.25 -> 3:1 (36+12), 0.17 -> 5:1 (40+8),
#   0.125 -> 7:1 (42+6), 0 -> fall back to proportional
IMC_FRACTION=0.17

# Regularization
#todo: change this part
WEIGHT_DECAY=0.04
WEIGHT_DECAY_END=0.1   # fine-tuning 后期不需要强正则


# DINO settings
MIM_LOSS_WEIGHT=1.0
MASK_RATIO=0.4
FREEZE_LAST_LAYER=0    # 去掉 freeze（ViT-S 不需要）

# KoLeo regularization
KOLEO_LOSS_WEIGHT=0.1

# Channel-mask consistency loss (Lmask) -- DINO-style CE.
# K student-masked CLS distributions are pulled to one teacher-full-panel
# CLS distribution per sample (K-to-1 pairing). EMA center buffer (separate
# from the main DINO center) prevents collapse. Teacher view = full panel;
# samples grouped by panel in the collator. Set MASK_LOSS_WEIGHT=0.0 to disable.
MASK_LOSS_WEIGHT=1.0
# Warmup not needed: pretrained backbone already gives meaningful full-panel
# vs. masked-panel teacher/student CLS, so mask CE has well-shaped gradient
# from iter 0. Empirical: with warmup=1 the mask raw loss dropped from 11.5
# to 4.3 in 1000 iters even at near-zero weight, showing the alignment work
# is done by DINO/MIM via the shared backbone, not by mask CE itself. Set 0
# to apply target weight from step 1 (formula falls back to constant weight
# when mask_warmup_epochs <= 0).
MASK_WARMUP_EPOCHS=0
# Per-view channel KEEP fraction (mirrors DINO's local_crops_scale=(0.05,0.4)
# convention: express the fraction-to-keep, not the fraction-to-mask). Each of
# the K student views samples its own keep ~ U[min, max], so K_s = floor(keep*C).
# At keep=0.05 with C=30, that floors to 1 channel -> rescued to MASK_MIN_KEPT_CHANNELS.
MASK_KEEP_MIN=0.05
MASK_KEEP_MAX=0.4
# Floor on student view channel count (includes the always-kept anchor like
# DAPI/DNA). 3 matches the DINO branch's 3-channel input width, so the model
# never sees a more severely truncated panel via the mask branch than via DINO.
MASK_MIN_KEPT_CHANNELS=3
# K student views per teacher view. Each gets an independent keep fraction and
# random channel subset. Loss = mean CE over K views.
# Temps + center momentum are shared with the main DINO loss (TEACHER_TEMP,
# STUDENT_TEMP defined below; center_momentum hardcoded to 0.9 to match DINO).
MASK_N_STUDENT_VIEWS=8
# Marker NAMES (comma-separated) that every student view must ALWAYS retain --
# the cell-skeleton / nuclear-DNA anchor channels (DAPI for CODEX marker_id=4,
# DNA for IMC marker_id=3). Names absent from a panel are silently ignored.
MASK_ALWAYS_KEEP_MARKERS="DAPI,DNA"

# Gradient accumulation
GRAD_ACCUM_STEPS=1

# Teacher temperature schedule (WARMUP_TEACHER_TEMP_EPOCHS 必须 <= EPOCHS)
#todo: change this part
WARMUP_TEACHER_TEMP=0.04
TEACHER_TEMP=0.04
WARMUP_TEACHER_TEMP_EPOCHS=1

# Run training (2x H200 distributed)
torchrun --nproc_per_node=2 --master-addr=127.0.0.1 --master-port=29500 finetune_kronos.py \
    --distributed \
    --bf16 \
    --data_path "${DATA_PATH[@]}" \
    --data_path_imc ${DATA_PATH_IMC} \
    --marker_metadata ${MARKER_METADATA} \
    --marker_metadata_imc ${MARKER_METADATA_IMC} \
    --output_dir ${OUTPUT_DIR} \
    --model_type ${MODEL_TYPE} \
    --pretrained_weights ${PRETRAINED_WEIGHTS} \
    --batch_size ${BATCH_SIZE} \
    --imc_fraction ${IMC_FRACTION} \
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
