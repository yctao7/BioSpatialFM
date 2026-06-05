"""
Fine-tuning script for KRONOS model using DINO self-supervised learning.
Now with proper MIM (Masked Image Modeling) support.
"""

import argparse
import math
import os
import sys
import time
import json
import random
from pathlib import Path

import torch    
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import csv
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from kronos import create_model_from_pretrained
from kronos.dino_head import DINOHead
from kronos.dino_loss import DINOLoss, MultiCropWrapper
from kronos.koleo_loss import KoLeoLoss
from kronos.data_augmentation import get_augmentation_pipeline
from kronos.dataset import (
    MultiplexImageDataset,
    MultiplexImageFolderDataset,
    MultiplexPatchDataset,
    collate_fn_multicrop,
    load_marker_metadata
)
import wandb


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Each batch contains exactly batch_size//2 samples from the primary dataset
    and batch_size//2 from the secondary dataset (IMC).
    Primary indices: [0, n_primary)
    Secondary indices: [n_primary, n_primary + n_secondary)
    """
    def __init__(self, n_primary, n_secondary, batch_size):
        assert batch_size % 2 == 0, "batch_size must be even for balanced sampling"
        self.n_primary = n_primary
        self.n_secondary = n_secondary
        self.half = batch_size // 2

    def __iter__(self):
        primary_idx = torch.randperm(self.n_primary).tolist()
        secondary_idx = (torch.randperm(self.n_secondary) + self.n_primary).tolist()
        n_batches = min(len(primary_idx), len(secondary_idx)) // self.half
        for i in range(n_batches):
            batch = (primary_idx[i * self.half:(i + 1) * self.half] +
                     secondary_idx[i * self.half:(i + 1) * self.half])
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return min(self.n_primary, self.n_secondary) // self.half


class DistributedBalancedSampler(torch.utils.data.Sampler):
    """
    Distributed-aware sampler that mixes primary (e.g. CODEX) and secondary
    (e.g. IMC) sources at a CONFIGURABLE ratio per batch. Each rank's epoch
    length is bounded by the primary shard so every primary patch is seen
    once per epoch; the secondary pool CYCLES with reshuffle to fill the
    secondary slots in every batch.

    Concretely with batch_size=48 and secondary_per_batch=8 (5:1 CODEX:IMC):
      - per batch: 40 primary + 8 secondary
      - per rank: primary_per_rank // 40 batches per epoch  (CODEX 690k / 2 / 40 = 8626)
      - secondary cycled with reshuffle each pass through its rank's shard
      - each unique IMC patch is seen ~(8626 * 8) / (n_secondary / world_size)
        times per epoch -- e.g. with IMC ~48k that's ~8626*8 / 24000 = 2.87x

    Primary indices: [0, n_primary). Secondary: [n_primary, n_primary + n_secondary).

    Why a FIXED ratio matters for cross-platform alignment losses:
      - Mask CE (current): unaffected by ratio so long as both panels appear in
        each batch (panel grouping happens inside DinoMaskCollator per-sample).
      - MMD (if added later): MMD = E[K(x,x')] + E[K(y,y')] - 2 E[K(x,y)]. The
        K_yy estimator's variance is O(1/n_secondary^2) per batch -- low IMC
        count -> noisy single-batch MMD. Each batch is still UNBIASED so the
        time-averaged estimator converges; just variance per step scales with
        secondary_per_batch. Enforcing a floor (>= ~4-8) keeps gradients stable.
        Variable ratios across batches don't bias MMD, only widen variance.
      - DANN (if added later): a domain classifier learns to distinguish CODEX
        vs IMC; an imbalanced batch (e.g. 40+8) lets the classifier hit ~83%
        accuracy by always predicting majority -- the GRL signal back to the
        encoder is therefore lopsided towards "fool classifier on CODEX". For
        DANN you'd want either pos_weight=primary/secondary in the classifier
        BCE, or a 1:1 batch (set imc_fraction=0.50). Variable ratios per batch
        are EXTRA bad for DANN because the classifier's update step size
        oscillates with batch composition, destabilizing adversarial training.

    In short: this fixed-ratio sampler is friendly to MMD as-is, but DANN would
    need either a pos_weighted classifier loss or imc_fraction=0.5 to be sound.
    """
    def __init__(self, n_primary, n_secondary, batch_size, secondary_per_batch,
                 rank, world_size, seed=0):
        assert 0 < secondary_per_batch < batch_size, \
            f"secondary_per_batch ({secondary_per_batch}) must be in (0, {batch_size})"
        self.n_primary = n_primary
        self.n_secondary = n_secondary
        self.batch_size = batch_size
        self.secondary_per_batch = int(secondary_per_batch)
        self.primary_per_batch = batch_size - self.secondary_per_batch
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

        # Each rank gets an equal-sized shard of primary; pad if needed.
        self.primary_per_rank = math.ceil(n_primary / world_size)
        # Secondary is also sharded across ranks (no duplicate samples per step).
        # If too small per rank to fill one batch alone, that's fine -- we cycle.
        self.secondary_per_rank = math.ceil(n_secondary / world_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Shuffle primary globally, partition by rank.
        primary_all = torch.randperm(self.n_primary, generator=g).tolist()
        primary_all += primary_all[:(self.primary_per_rank * self.world_size - len(primary_all))]
        primary_shard = primary_all[self.rank * self.primary_per_rank:
                                    (self.rank + 1) * self.primary_per_rank]

        # Shuffle secondary globally, partition by rank.
        secondary_all = (torch.randperm(self.n_secondary, generator=g)
                         + self.n_primary).tolist()
        secondary_all += secondary_all[:(self.secondary_per_rank * self.world_size - len(secondary_all))]
        secondary_shard = secondary_all[self.rank * self.secondary_per_rank:
                                        (self.rank + 1) * self.secondary_per_rank]

        # Iter count bounded by primary -- secondary cycles.
        n_batches = len(primary_shard) // self.primary_per_batch
        secondary_needed = n_batches * self.secondary_per_batch
        # Cycle secondary with fresh reshuffles each pass through the shard so
        # consecutive batches don't see the same secondary samples in the same order.
        g2 = torch.Generator()
        g2.manual_seed(self.seed + self.epoch * 7919 + self.rank)
        secondary_pool = []
        while len(secondary_pool) < secondary_needed:
            order = torch.randperm(len(secondary_shard), generator=g2).tolist()
            secondary_pool.extend(secondary_shard[i] for i in order)
        secondary_pool = secondary_pool[:secondary_needed]

        for i in range(n_batches):
            batch = (primary_shard[i * self.primary_per_batch:(i + 1) * self.primary_per_batch] +
                     secondary_pool[i * self.secondary_per_batch:(i + 1) * self.secondary_per_batch])
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.primary_per_rank // self.primary_per_batch


class LossLogger:
    """Logger for tracking and saving loss curves."""
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.loss_history = []
        self.log_file = self.log_dir / 'loss_curves.csv'

        # Initialize CSV file with headers
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'iteration', 'total_loss', 'cls_loss', 'mim_loss', 'lr', 'wd'])

    def log(self, epoch, iteration, total_loss, cls_loss, mim_loss, lr, wd):
        """Log loss values."""
        entry = {
            'epoch': epoch,
            'iteration': iteration,
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'mim_loss': mim_loss,
            'lr': lr,
            'wd': wd
        }
        self.loss_history.append(entry)

        # Append to CSV file
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, iteration, total_loss, cls_loss, mim_loss, lr, wd])

    def save_epoch_summary(self, epoch, train_stats):
        """Save epoch summary to a separate file."""
        summary_file = self.log_dir / 'epoch_summary.csv'

        # Check if file exists to write header
        file_exists = summary_file.exists()

        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'avg_total_loss', 'avg_cls_loss', 'avg_mim_loss', 'avg_lr', 'avg_wd'])

            writer.writerow([
                epoch,
                train_stats.get('loss', 0),
                train_stats.get('cls_loss', 0),
                train_stats.get('mim_loss', 0),
                train_stats.get('lr', 0),
                train_stats.get('wd', 0)
            ])


def get_args_parser():
    parser = argparse.ArgumentParser('KRONOS Fine-tuning', add_help=False)
    
    # Model parameters
    parser.add_argument('--model_type', default='vits16', type=str, 
                        choices=['vits16', 'vitl16'],
                        help='Model architecture')
    parser.add_argument('--pretrained_weights', default='hf_hub:MahmoodLab/kronos', 
                        type=str, help='Path to pretrained weights')
    parser.add_argument('--token_overlap', action='store_true',
                        help='Use token overlap (stride_size=8)')
    parser.add_argument('--drop_path_rate', default=0.3, type=float,
                        help='Stochastic depth drop path rate (0.3 for pretraining, 0.1 for fine-tuning)')
    
    # Training parameters
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Per-GPU batch size')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs')
    parser.add_argument('--lr', default=0.004, type=float,
                        help='Learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help='Number of warmup epochs')
    parser.add_argument('--weight_decay', default=0.04, type=float,
                        help='Weight decay')
    parser.add_argument('--weight_decay_end', default=0.1, type=float,
                        help='Final weight decay')
    parser.add_argument('--clip_grad', default=3.0, type=float,
                        help='Gradient clipping')
    parser.add_argument('--freeze_last_layer', default=1, type=int,
                        help='Number of epochs to freeze last layer')
    
    # DINO parameters
    parser.add_argument('--out_dim', default=65536, type=int,
                        help='Dimensionality of DINO head output')
    parser.add_argument('--norm_last_layer', action='store_true',
                        help='Whether to normalize the last layer of DINO head')
    parser.add_argument('--momentum_teacher', default=0.992, type=float,
                        help='EMA parameter for teacher update')
    parser.add_argument('--use_bn_in_head', action='store_true',
                        help='Whether to use batch norm in DINO head')
    
    # Temperature parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help='Initial teacher temperature')
    parser.add_argument('--teacher_temp', default=0.07, type=float,
                        help='Final teacher temperature')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
                        help='Number of epochs for teacher temperature warmup')
    parser.add_argument('--student_temp', default=0.1, type=float,
                        help='Student temperature')
    parser.add_argument('--mim_loss_weight', default=1.0, type=float,
                        help='Weight for MIM loss (relative to CLS loss)')
    parser.add_argument('--koleo_loss_weight', default=0.1, type=float,
                        help='Weight for KoLeo regularization loss (0.0 to disable)')
    parser.add_argument('--grad_accum_steps', default=1, type=int,
                        help='Gradient accumulation steps to simulate larger batch size')
    
    # MIM parameters (NEW)
    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Ratio of patches to mask for MIM (0.0 to disable MIM)')
    parser.add_argument('--mask_global_crops_only', action='store_true', default=True,
                        help='Only apply masking to global crops (recommended)')

    # Channel-mask consistency loss (Lmask) parameters
    # DINO-style CE between K student-masked CLS distributions and one
    # teacher-full-panel CLS distribution per sample (K-to-1 pairing). Teacher
    # view is always the full marker panel; samples with different panel sizes
    # (CODEX/IMC) are grouped by panel in the collator and a single forward
    # is run per side per view via xFormers BlockDiagonal packing.
    parser.add_argument('--mask_loss_weight', default=0.0, type=float,
                        help='Weight for channel-mask consistency loss (0.0 disables it)')
    parser.add_argument('--mask_warmup_epochs', default=1, type=int,
                        help='Linearly warm up mask_loss_weight from 0 over this many epochs')
    parser.add_argument('--mask_keep_min', default=0.05, type=float,
                        help='Min per-view channel KEEP fraction (mirrors DINO '
                             'local_crops_scale convention: fraction-to-keep, not '
                             'fraction-to-mask). Each of K student views samples its '
                             'own keep ~ U[min, max], then K_s = floor(keep * C).')
    parser.add_argument('--mask_keep_max', default=0.4, type=float,
                        help='Max per-view channel KEEP fraction.')
    parser.add_argument('--mask_min_kept_channels', default=3, type=int,
                        help='Floor on student view channel count (includes always-kept '
                             'anchors like DAPI/DNA). Default 3 mirrors the DINO '
                             'branch input width.')
    parser.add_argument('--mask_n_student_views', default=2, type=int,
                        help='Number of independent masked student views per teacher '
                             'view (K). Loss = mean over K views; teacher forward is '
                             'run only once.')
    parser.add_argument('--mask_always_keep_markers', default='DAPI', type=str,
                        help='Comma-separated marker NAMES that must always be PRESENT '
                             'in the student view in Lmask (mirrors MarkerSelection in '
                             'the DINO branch -- DAPI is the nuclear/structural anchor '
                             'so removing it makes embedding alignment ill-posed). The '
                             'mask ratio r still applies; non-forced channels are sampled '
                             'uniformly from the rest. Set to empty string to disable.')
    
    # Augmentation parameters
    parser.add_argument('--global_crops_scale', default=(0.48, 1.0), type=tuple,
                        help='Scale range for global crops')
    parser.add_argument('--local_crops_scale', default=(0.16, 0.48), type=tuple,
                        help='Scale range for local crops')
    parser.add_argument('--local_crops_number', default=8, type=int,
                        help='Number of local crops')
    parser.add_argument('--global_crops_size', default=224, type=int,
                        help='Size of global crops')
    parser.add_argument('--local_crops_size', default=96, type=int,
                        help='Size of local crops')
    
    # Dataset parameters
    parser.add_argument('--imc_fraction', default=0.17, type=float,
                        help='Fraction of each batch that is IMC samples (the rest is '
                             'CODEX). Default 0.17 (~1/6) gives 5:1 CODEX:IMC. '
                             'secondary_per_batch = round(batch_size * imc_fraction). '
                             'Examples: 0.50 -> 1:1, 0.25 -> 3:1, 0.17 -> 5:1, '
                             '0.125 -> 7:1. Set to 0 to fall back to plain '
                             'DistributedSampler (proportional, ~2% IMC per batch).')
    parser.add_argument('--data_path', required=True, type=str, nargs='+',
                        help='Path(s) to primary dataset(s) (e.g. CODEX, islet). '
                             'Multiple paths are concatenated. Each path is walked '
                             'recursively for *.h5 patches, so both flat dirs (CODEX) '
                             'and nested dirs (islet) work.')
    parser.add_argument('--data_path_imc', default='', type=str,
                        help='Path to secondary dataset (e.g. IMC); merged with data_path if provided')
    parser.add_argument('--dataset_type', default='patch', type=str,
                        choices=['folder', 'patch', 'list'],
                        help='Type of dataset')
    parser.add_argument('--marker_metadata', default='tutorials/codex_dataset/dataset/marker_info_with_metadata.csv', type=str,
                        help='Path to marker metadata CSV file')
    parser.add_argument('--marker_metadata_imc', default='', type=str,
                        help='Path to IMC marker metadata CSV; merged with --marker_metadata if provided')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='Number of data loading workers')
    parser.add_argument('--bf16', action='store_true',
                        help='Use bfloat16 autocast instead of float16+GradScaler. '
                             'bf16 has fp32-equivalent dynamic range so no loss scaling is '
                             'needed. Native on Ampere/Hopper (A100/H100/H200).')
    
    # Checkpoint parameters
    parser.add_argument('--output_dir', default='./output/finetune', type=str,
                        help='Path to save checkpoints')
    parser.add_argument('--saveckp_freq', default=1, type=int,
                        help='Save checkpoint every x epochs')
    parser.add_argument('--resume', default='', type=str,
                        help='Path to checkpoint to resume from')
    
    # WandB parameters
    parser.add_argument('--wandb_project', default='', type=str,
                        help='WandB project name (empty to disable)')
    parser.add_argument('--wandb_run_name', default='', type=str,
                        help='WandB run name')

    # Distributed training parameters
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='Local rank for distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='URL for distributed training setup')
    
    return parser


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                   data_loader, optimizer, lr_schedule, wd_schedule,
                   momentum_schedule, epoch, fp16_scaler, args, loss_logger=None,
                   use_wandb=False, koleo_loss_fn=None, mask_loss_fn=None):
    """Train for one epoch with optional gradient accumulation and KoLeo loss."""
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}/{args.epochs}]'

    grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
    koleo_weight = getattr(args, 'koleo_loss_weight', 0.0)
    mask_weight_target = getattr(args, 'mask_loss_weight', 0.0)
    mask_warmup_epochs = max(0, getattr(args, 'mask_warmup_epochs', 0))
    use_mask_loss = mask_weight_target > 0.0

    # zero_grad before the loop; will re-zero after each optimizer step
    optimizer.zero_grad()

    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # The combined dino+mask collator returns a dict; the legacy collator returns a tuple.
        if isinstance(batch, dict):
            images = batch['dino_crops']
            marker_ids = batch['dino_marker_ids']
            mask_groups = batch['mask_groups']  # list of {teacher_imgs, teacher_ids, student_views=[{imgs,ids,K_s,r}], C}
        else:
            images, marker_ids = batch
            mask_groups = None

        # Update LR and WD every micro-step
        it_global = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it_global]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it_global]

        # Linearly ramp the mask-loss weight from 0 to its target across mask_warmup_epochs.
        # At fine-tuning time the EMA teacher is initialized from a pretrained checkpoint, so
        # the warmup is just a short safety margin to let the teacher adapt to full-panel input.
        if use_mask_loss and mask_warmup_epochs > 0:
            mask_weight = mask_weight_target * min(
                1.0, (epoch + (it / max(1, len(data_loader)))) / mask_warmup_epochs
            )
        else:
            mask_weight = mask_weight_target

        # Move to GPU. marker_ids may be List[List[Tensor]] (multi-crop, per-sample)
        # or List[Tensor] (single-image legacy path).
        if isinstance(images, list):
            images = [im.cuda(non_blocking=True) for im in images]
            if marker_ids and isinstance(marker_ids[0], list):
                marker_ids = [
                    [m.cuda(non_blocking=True) for m in per_crop]
                    for per_crop in marker_ids
                ]
            else:
                marker_ids = [m.cuda(non_blocking=True) for m in marker_ids]
        else:
            images = images.cuda(non_blocking=True)
            marker_ids = marker_ids.cuda(non_blocking=True)

        if use_mask_loss and mask_groups is not None:
            for grp in mask_groups:
                grp['teacher_imgs'] = grp['teacher_imgs'].cuda(non_blocking=True)
                grp['teacher_ids'] = grp['teacher_ids'].cuda(non_blocking=True)
                for v in grp['student_views']:
                    v['imgs'] = v['imgs'].cuda(non_blocking=True)
                    v['ids'] = v['ids'].cuda(non_blocking=True)

        # Forward pass. autocast enabled for both fp16 (with scaler) and bf16 (no scaler).
        amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
        amp_enabled = args.bf16 or (fp16_scaler is not None)
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
            teacher_ret = teacher(images[:2], marker_ids=marker_ids[:2], is_student=False)
            student_ret = student(images, marker_ids=marker_ids, is_student=True)

            total_loss, cls_loss, mim_loss = dino_loss(
                student_output=student_ret["x_norm_clstoken"],
                teacher_output=teacher_ret["x_norm_clstoken"],
                student_patch_out=student_ret.get("x_norm_patchtokens"),
                teacher_patch_out=teacher_ret.get("x_norm_patchtokens"),
                masks=student_ret.get("masks"),
                epoch=epoch
            )

            # KoLeo loss on backbone CLS tokens of global crops (student only)
            koleo_loss_val = torch.tensor(0.0, device=total_loss.device)
            if koleo_loss_fn is not None and koleo_weight > 0:
                backbone_cls = student_ret.get("backbone_cls_tokens")
                if backbone_cls is not None:
                    koleo_loss_val = koleo_loss_fn(backbone_cls)
                    total_loss = total_loss + koleo_weight * koleo_loss_val

            # Channel-mask consistency loss (Lmask) -- DINO-style CE.
            #
            # Single teacher forward (full panel, no_grad) for all panel groups,
            # K student forwards (one per masked-view realization). All forwards
            # go through MultiCropWrapper.forward(..., mask_branch=True), which
            # routes to backbone.forward_features_list -- xFormers BlockDiagonal
            # packs all panel groups into a single attention call per forward.
            # K-to-1 pairing per sample: K student distributions all match the
            # same teacher target distribution.
            mask_loss_val = torch.tensor(0.0, device=total_loss.device)
            mask_koleo_val = torch.tensor(0.0, device=total_loss.device)
            if use_mask_loss and mask_weight > 0.0 and mask_groups and mask_loss_fn is not None:
                teacher_x = [grp['teacher_imgs'] for grp in mask_groups]
                teacher_m = [grp['teacher_ids'] for grp in mask_groups]
                with torch.no_grad():
                    mt_ret = teacher(teacher_x, marker_ids=teacher_m, mask_branch=True)
                    z_t = mt_ret["x_norm_clstoken"].detach()  # [N, head_out_dim]

                # K student forwards, each producing [N, head_out_dim] (head) and
                # [N, embed_dim] (pre-head backbone CLS, used for KoLeo).
                n_views = len(mask_groups[0]['student_views']) if mask_groups else 0
                student_z_list = []
                student_backbone_cls_list = []
                for k in range(n_views):
                    s_x = [grp['student_views'][k]['imgs'] for grp in mask_groups]
                    s_m = [grp['student_views'][k]['ids']  for grp in mask_groups]
                    ms_ret = student(s_x, marker_ids=s_m, mask_branch=True)
                    student_z_list.append(ms_ret["x_norm_clstoken"])
                    if koleo_loss_fn is not None and koleo_weight > 0:
                        student_backbone_cls_list.append(ms_ret["backbone_cls_tokens"])

                mask_loss_val = mask_loss_fn(student_z_list, z_t)
                total_loss = total_loss + mask_weight * mask_loss_val

                # KoLeo regularization on the mask-branch backbone CLS (mirrors
                # DINOv2 practice of regularizing every student-CLS-producing branch).
                # Concatenate across K views so KoLeo searches NN over more samples.
                if student_backbone_cls_list:
                    mask_backbone_cls = torch.cat(student_backbone_cls_list, dim=0)
                    mask_koleo_val = koleo_loss_fn(mask_backbone_cls)
                    total_loss = total_loss + koleo_weight * mask_koleo_val

        if not torch.isfinite(total_loss):
            print(f"Loss is {total_loss}, stopping training")
            sys.exit(1)

        # Backward (scale by grad_accum_steps to keep effective loss magnitude)
        scaled_loss = total_loss / grad_accum_steps
        if fp16_scaler is None:
            scaled_loss.backward()
        else:
            fp16_scaler.scale(scaled_loss).backward()

        # Optimizer step every grad_accum_steps micro-steps (or at end of epoch)
        is_last_iter = (it + 1 == len(data_loader))
        if (it + 1) % grad_accum_steps == 0 or is_last_iter:
            if fp16_scaler is None:
                if args.clip_grad:
                    clip_gradients(student, args.clip_grad)
                cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
                optimizer.step()
            else:
                if args.clip_grad:
                    fp16_scaler.unscale_(optimizer)
                    clip_gradients(student, args.clip_grad)
                cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            optimizer.zero_grad()

            # EMA teacher update only on actual optimizer steps
            with torch.no_grad():
                m = momentum_schedule[it_global]
                for param_q, param_k in zip(student.parameters(), teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # Logging (per micro-step)
        torch.cuda.synchronize()
        metric_logger.update(loss=total_loss.item())
        metric_logger.update(cls_loss=cls_loss.item())
        metric_logger.update(mim_loss=mim_loss.item())
        if koleo_weight > 0:
            metric_logger.update(koleo_loss=koleo_loss_val.item())
        if use_mask_loss:
            metric_logger.update(mask_loss=mask_loss_val.item())
            metric_logger.update(mask_w=mask_weight)
            if koleo_loss_fn is not None and koleo_weight > 0:
                metric_logger.update(mask_koleo_loss=mask_koleo_val.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        # Log to CSV
        if loss_logger is not None:
            loss_logger.log(
                epoch=epoch,
                iteration=it_global,
                total_loss=total_loss.item(),
                cls_loss=cls_loss.item(),
                mim_loss=mim_loss.item(),
                lr=optimizer.param_groups[0]["lr"],
                wd=optimizer.param_groups[0]["weight_decay"]
            )

        # Log to WandB
        if use_wandb:
            log_data = {
                'iter_loss': total_loss.item(),
                'iter_cls_loss': cls_loss.item(),
                'iter_mim_loss': mim_loss.item(),
                'iter_lr': optimizer.param_groups[0]["lr"],
                'iter_wd': optimizer.param_groups[0]["weight_decay"],
                'epoch': epoch,
                'global_step': it_global,
            }
            if koleo_weight > 0:
                log_data['iter_koleo_loss'] = koleo_loss_val.item()
            if use_mask_loss:
                log_data['iter_mask_loss'] = mask_loss_val.item()
                log_data['iter_mask_weight'] = mask_weight
                if koleo_loss_fn is not None and koleo_weight > 0:
                    log_data['iter_mask_koleo_loss'] = mask_koleo_val.item()
            wandb.log(log_data)

        if it % 50 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            current_wd = optimizer.param_groups[0]["weight_decay"]
            mask_str = (f", Mask: {mask_loss_val.item():.4f}@w={mask_weight:.3f}"
                        if use_mask_loss else "")
            print(f"  [Iter {it:4d}] Loss: {total_loss.item():.4f} "
                  f"(CLS: {cls_loss.item():.4f}, MIM: {mim_loss.item():.4f}, "
                  f"KoLeo: {koleo_loss_val.item():.4f}{mask_str}) | "
                  f"LR: {current_lr:.8f} | WD: {current_wd:.6f}")

    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def clip_gradients(model, clip):
    """Clip gradients to prevent explosion."""
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, student, freeze_last_layer):
    """Zero gradients on DINO head's last layer for the first freeze_last_layer epochs."""
    if epoch >= freeze_last_layer:
        return
    for n, p in student.named_parameters():
        if "last_layer" in n:
            p.grad = None


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0):
    """Cosine learning rate schedule with warmup."""
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class MetricLogger:
    """Simple metric logger."""
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if name == 'lr':
                loss_str.append(f"{name}: {meter.avg:.8f}")  
            else:
                loss_str.append(f"{name}: {meter.avg:.4f}")
        return self.delimiter.join(loss_str)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = AverageMeter()
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time:.4f}',
        ]
        log_msg = self.delimiter.join(log_msg)
        
        for obj in iterable:
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.avg * (len(iterable) - i)
                eta_string = str(int(eta_seconds))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=iter_time.avg
                ))
            i += 1
            end = time.time()


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    @property
    def global_avg(self):
        return self.avg
    
    def synchronize_between_processes(self):
        if not dist.is_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = t[0]
        self.count = t[1]
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth'):
    """Save checkpoint."""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(checkpoint_path, student, teacher, optimizer, fp16_scaler,
                    dino_loss=None, mask_loss_fn=None):
    """Load checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    student.load_state_dict(checkpoint['student'])
    teacher.load_state_dict(checkpoint['teacher'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if fp16_scaler is not None and 'fp16_scaler' in checkpoint:
        fp16_scaler.load_state_dict(checkpoint['fp16_scaler'])
    if dino_loss is not None and 'dino_loss' in checkpoint:
        dino_loss.load_state_dict(checkpoint['dino_loss'])
    # MaskCEloss center buffer (independent of dino_loss center). Without this
    # the mask center resets to 0 on resume, causing the first ~10-20 iters of
    # mask CE to be inflated until the EMA re-warms.
    if mask_loss_fn is not None and 'mask_loss' in checkpoint and checkpoint['mask_loss'] is not None:
        mask_loss_fn.load_state_dict(checkpoint['mask_loss'])
        print(f"  mask_loss center buffer restored (norm={mask_loss_fn.center.norm().item():.4f})")

    return checkpoint['epoch']


def main(args):
    # Setup distributed training
    if args.distributed:
        # torchrun sets LOCAL_RANK/RANK/WORLD_SIZE as env vars; read them here
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        rank = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        args.local_rank = local_rank
        args.world_size = world_size
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://',
                               world_size=world_size, rank=rank)
        print(f"Distributed training enabled: rank {rank}/{world_size} on GPU {local_rank}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    use_wandb = WANDB_AVAILABLE and args.wandb_project and args.local_rank == 0
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            config=vars(args),
            dir=args.output_dir,
        )
        print(f"WandB initialized: project={args.wandb_project}, run={wandb.run.name}")
    elif args.wandb_project and not WANDB_AVAILABLE:
        print("Warning: --wandb_project specified but wandb is not installed.")

    # Create loss logger
    loss_logger = None
    if args.local_rank == 0:
        log_dir = os.path.join(args.output_dir, 'logs')
        loss_logger = LossLogger(log_dir)
        print(f"Loss curves will be saved to {log_dir}")

    # Save args
    if args.local_rank == 0:
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        wandb.init(project="KRONOS-Finetune", config=vars(args))
    
    # Load marker metadata if provided
    if args.marker_metadata:
        marker_metadata = load_marker_metadata(args.marker_metadata)
    else:
        raise ValueError("Marker metadata file must be provided.")
    if args.marker_metadata_imc:
        marker_metadata_imc = load_marker_metadata(args.marker_metadata_imc)
        overlap = set(marker_metadata.keys()) & set(marker_metadata_imc.keys())
        if overlap:
            print(f"Warning: {len(overlap)} overlapping markers in metadata files: {overlap}")
        marker_metadata = {**marker_metadata, **marker_metadata_imc}
        print(f"Merged marker metadata: {len(marker_metadata)} markers total")
    
    # Setup data augmentation. When mask_loss_weight > 0 we use the combined pipeline
    # that emits BOTH the standard DINO multicrop output AND the mask-consistency views
    # from a single h5 read.
    use_mask_loss = args.mask_loss_weight > 0.0
    transform = get_augmentation_pipeline(
        marker_metadata,
        augmentation_type=('dino+mask' if use_mask_loss else 'dino'),
        global_crops_size=args.global_crops_size,
        local_crops_size=args.local_crops_size,
        local_crops_number=args.local_crops_number,
        global_crops_scale=args.global_crops_scale,
        local_crops_scale=args.local_crops_scale,
    )
    
    # Create dataset
    def _make_dataset(data_path, recursive=False):
        if args.dataset_type == 'folder':
            return MultiplexImageFolderDataset(data_root=data_path, transform=transform)
        elif args.dataset_type == 'patch':
            return MultiplexPatchDataset(patch_dir=data_path, transform=transform, recursive=recursive)
        else:
            raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    # data_path may be a single string (legacy) or a list of paths (multi-source primary).
    # Use recursive=True so flat dirs (e.g. CODEX) and nested dirs (e.g. islet) both work.
    primary_paths = args.data_path if isinstance(args.data_path, list) else [args.data_path]
    primary_datasets = [_make_dataset(p, recursive=True) for p in primary_paths]
    for p, d in zip(primary_paths, primary_datasets):
        print(f"Primary source {p}: {len(d)} patches")
    if len(primary_datasets) == 1:
        dataset = primary_datasets[0]
    else:
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset(primary_datasets)
    n_primary = len(dataset)
    n_imc = 0
    if args.data_path_imc:
        from torch.utils.data import ConcatDataset
        # New IMC patches (extracted from hpapdata_comb via patch_extractor.py default
        # codex mode) are written to a single flat directory with canonical marker
        # names matching the IMC CSV. If you ever switch back to a nested IMC layout,
        # bump this to recursive=True.
        dataset_imc = _make_dataset(args.data_path_imc, recursive=False)
        n_imc = len(dataset_imc)
        print(f"Mixed dataset: primary={n_primary}, imc={n_imc}, total={n_primary + n_imc}")
        dataset = ConcatDataset([dataset, dataset_imc])

    print(f"Dataset size: {len(dataset)}")

    # Create dataloader
    if use_mask_loss:
        from kronos.dataset import DinoMaskCollator
        # Resolve marker NAMES (e.g. "DAPI") to marker_ids via marker_metadata.
        always_keep_marker_ids = []
        raw = (args.mask_always_keep_markers or '').strip()
        if raw:
            for name in [n.strip() for n in raw.split(',') if n.strip()]:
                meta = marker_metadata.get(name)
                if meta is None:
                    print(f"  [Lmask] Warning: --mask_always_keep_markers includes "
                          f"'{name}' which is not in the marker metadata; ignoring.")
                else:
                    mid = int(meta.get('marker_id', 0))
                    always_keep_marker_ids.append(mid)
                    print(f"  [Lmask] always-keep '{name}' (marker_id={mid}) in "
                          f"student view (anchor channel)")
        collate_fn = DinoMaskCollator(
            keep_min=args.mask_keep_min,
            keep_max=args.mask_keep_max,
            always_keep_marker_ids=always_keep_marker_ids,
            n_student_views=args.mask_n_student_views,
            min_kept_channels=args.mask_min_kept_channels,
        )
    else:
        collate_fn = collate_fn_multicrop

    if args.distributed:
        # If a secondary IMC source is configured, use the ratio-aware balanced
        # sampler so each batch has a guaranteed number of IMC samples (otherwise
        # plain DistributedSampler samples proportional to dataset sizes, which
        # gives ~1.7% IMC per batch given 690k CODEX vs 12k IMC -- mask CE on
        # IMC would be severely under-trained).
        if args.data_path_imc and n_imc > 0 and args.imc_fraction > 0:
            rank = int(os.environ.get('RANK', args.local_rank))
            ws = int(os.environ.get('WORLD_SIZE', args.world_size))
            secondary_per_batch = max(1, int(round(args.batch_size * args.imc_fraction)))
            assert secondary_per_batch < args.batch_size, \
                f"imc_fraction={args.imc_fraction} too high for batch={args.batch_size}"
            batch_sampler = DistributedBalancedSampler(
                n_primary=n_primary,
                n_secondary=n_imc,
                batch_size=args.batch_size,
                secondary_per_batch=secondary_per_batch,
                rank=rank,
                world_size=ws,
            )
            print(f"DistributedBalancedSampler: {len(batch_sampler)} batches/rank/epoch "
                  f"({batch_sampler.primary_per_batch} CODEX + "
                  f"{batch_sampler.secondary_per_batch} IMC per batch "
                  f"= {batch_sampler.secondary_per_batch / args.batch_size * 100:.0f}% IMC, "
                  f"IMC cycles ~{(len(batch_sampler) * batch_sampler.secondary_per_batch) / max(1, batch_sampler.secondary_per_rank):.2f}x/epoch)")
            data_loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )
            sampler = batch_sampler  # for set_epoch in train loop
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            data_loader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=collate_fn,
            )
    elif args.data_path_imc:
        # Balanced: exactly half CODEX, half IMC per batch
        batch_sampler = BalancedBatchSampler(n_primary, n_imc, args.batch_size)
        print(f"BalancedBatchSampler: {len(batch_sampler)} batches/epoch "
              f"({args.batch_size // 2} CODEX + {args.batch_size // 2} IMC per batch)")
        data_loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    else:
        data_loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
    
    # Create student and teacher models
    print("Creating student model...")
    student_backbone, precision, embed_dim = create_model_from_pretrained(
        checkpoint_path=args.pretrained_weights,
        cfg={"model_type": args.model_type, "token_overlap": args.token_overlap,
             "drop_path_rate": args.drop_path_rate}
    )

    print("Creating teacher model...")
    teacher_backbone, _, _ = create_model_from_pretrained(
        checkpoint_path=args.pretrained_weights,
        cfg={"model_type": args.model_type, "token_overlap": args.token_overlap,
             "drop_path_rate": args.drop_path_rate}
    )
    
    # Create DINO heads
    student_head = DINOHead(
        in_dim=embed_dim,
        out_dim=args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    )
    teacher_head = DINOHead(
        in_dim=embed_dim,
        out_dim=args.out_dim,
        use_bn=args.use_bn_in_head,
    )
    
    # Wrap with MultiCropWrapper (now with MIM support)
    print(f"MIM enabled: mask_ratio={args.mask_ratio}, global_crops_only={args.mask_global_crops_only}")
    student = MultiCropWrapper(
        student_backbone, 
        student_head,
        mask_ratio=args.mask_ratio,
        mask_global_crops_only=args.mask_global_crops_only
    )
    teacher = MultiCropWrapper(
        teacher_backbone, 
        teacher_head,
        mask_ratio=0.0,  # Teacher never uses masking
        mask_global_crops_only=True
    )
    
    # Move to GPU
    student = student.cuda()
    teacher = teacher.cuda()
    
    # Teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict(), strict=False)
    
    # Teacher is not trained
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Wrap with DDP (teacher excluded: no grad, updated via EMA only)
    if args.distributed:
        student = DDP(student, device_ids=[args.local_rank], find_unused_parameters=True)
        teacher_without_ddp = teacher
    else:
        teacher_without_ddp = teacher
    
    print(f"Student and Teacher are ready. Embedding dim: {embed_dim}")
    
    # Create KoLeo loss (optional)
    koleo_loss_fn = None
    if getattr(args, 'koleo_loss_weight', 0.0) > 0:
        koleo_loss_fn = KoLeoLoss().cuda()
        print(f"KoLeo loss enabled with weight={args.koleo_loss_weight}")

    # Create channel-mask consistency loss (DINO-style CE; optional).
    # Reuses DINO's --teacher_temp / --student_temp; center_momentum hardcoded
    # to 0.9 to match DINO's center EMA convention. The mask center buffer is
    # SEPARATE from the main DINO center buffer (different output distributions:
    # full panel vs. multicrop), but the EMA hyperparameters are shared.
    mask_loss_fn = None
    if use_mask_loss:
        from kronos.mask_loss import MaskCEloss
        mask_loss_fn = MaskCEloss(
            out_dim=args.out_dim,
            teacher_temp=args.teacher_temp,
            student_temp=args.student_temp,
            center_momentum=0.9,
        ).cuda()
        print(f"Mask CE loss enabled: weight={args.mask_loss_weight} "
              f"warmup_epochs={args.mask_warmup_epochs} "
              f"n_views={args.mask_n_student_views} "
              f"keep~U[{args.mask_keep_min},{args.mask_keep_max}] "
              f"min_kept={args.mask_min_kept_channels} "
              f"(temps shared with DINO: teacher={args.teacher_temp}, "
              f"student={args.student_temp})")

    if getattr(args, 'grad_accum_steps', 1) > 1:
        print(f"Gradient accumulation enabled: {args.grad_accum_steps} steps "
              f"(effective batch size = {args.batch_size * args.grad_accum_steps})")

    # Create DINO loss
    dino_loss = DINOLoss(
        out_dim=args.out_dim,
        ncrops=2 + args.local_crops_number,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
        student_temp=args.student_temp,
        mim_loss_weight=args.mim_loss_weight,
    ).cuda()
    
    # Setup optimizer
    params_groups = [
        {'params': [p for p in student.parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(params_groups, lr=args.lr, weight_decay=args.weight_decay)

    effective_batch = args.batch_size * args.world_size
    if effective_batch < 32:
        base_lr = args.lr  
        print(f"Small batch ({effective_batch}), using LR without scaling: {base_lr}")
    else:
        base_lr = args.lr * effective_batch / 128.
        print(f"Large batch ({effective_batch}), scaling LR: {base_lr}")
    
    # Setup learning rate schedule
    lr_schedule = cosine_scheduler(
        base_lr,
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=0  # warmup 从 0 开始，正确升到 base_lr
    )
    
    # Setup weight decay schedule
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    
    # Setup momentum schedule for teacher
    momentum_schedule = cosine_scheduler(
        args.momentum_teacher,
        1,
        args.epochs, len(data_loader)
    )
    
    # Setup mixed precision training. bf16 wins on Ampere/Hopper: same throughput
    # as fp16 but full fp32 dynamic range, so no GradScaler / unscale dance.
    fp16_scaler = None
    if args.bf16:
        precision = torch.bfloat16
    elif precision == torch.float16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        # Checkpoint stores the epoch that was JUST COMPLETED (e.g. 'epoch': 7
        # after epoch 7 finished). Resuming, we want to start at epoch 8, so +1.
        # Without the +1, epoch 7 would re-run with stale lr_schedule indexing.
        start_epoch = load_checkpoint(
            args.resume, student, teacher, optimizer, fp16_scaler,
            dino_loss=dino_loss, mask_loss_fn=mask_loss_fn,
        ) + 1
        print(f"Resumed: starting from epoch {start_epoch}")
    
    print(f"Starting training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            # When DataLoader uses a batch_sampler (DistributedBalancedSampler),
            # the per-epoch shuffle hook lives on batch_sampler, not sampler.
            sampler_with_set_epoch = (
                data_loader.batch_sampler
                if hasattr(data_loader.batch_sampler, 'set_epoch')
                else data_loader.sampler
            )
            sampler_with_set_epoch.set_epoch(epoch)
        
        # Train one epoch
        train_stats = train_one_epoch(
            student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule,
            momentum_schedule, epoch, fp16_scaler, args, loss_logger,
            use_wandb=use_wandb, koleo_loss_fn=koleo_loss_fn,
            mask_loss_fn=mask_loss_fn,
        )

        # Save epoch summary
        if args.local_rank == 0 and loss_logger is not None:
            loss_logger.save_epoch_summary(epoch, train_stats)

        # Log epoch summary to WandB
        if use_wandb:
            epoch_log = {
                'epoch_loss': train_stats.get('loss', 0),
                'epoch_cls_loss': train_stats.get('cls_loss', 0),
                'epoch_mim_loss': train_stats.get('mim_loss', 0),
                'epoch': epoch,
            }
            if getattr(args, 'koleo_loss_weight', 0.0) > 0:
                epoch_log['epoch_koleo_loss'] = train_stats.get('koleo_loss', 0)
            wandb.log(epoch_log)

        # Save checkpoint
        if args.local_rank == 0:
            if (epoch + 1) % args.saveckp_freq == 0 or epoch == args.epochs - 1:
                save_dict = {
                    'student': student.state_dict(),
                    'teacher': teacher.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'dino_loss': dino_loss.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if fp16_scaler is not None:
                    save_dict['fp16_scaler'] = fp16_scaler.state_dict()
                # Persist the mask-CE center buffer so resume doesn't lose its
                # accumulated EMA. Stored as None when mask loss was not enabled.
                save_dict['mask_loss'] = (
                    mask_loss_fn.state_dict() if mask_loss_fn is not None else None
                )
                
                save_checkpoint(
                    save_dict,
                    os.path.join(args.output_dir, f'checkpoint_{epoch:04d}.pth')
                )
                save_checkpoint(
                    save_dict,
                    os.path.join(args.output_dir, 'checkpoint_latest.pth')
                )
    
    total_time = time.time() - start_time
    print(f'Training time: {total_time/3600:.2f} hours')

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('KRONOS Fine-tuning', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)