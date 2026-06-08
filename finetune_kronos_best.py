"""
Fine-tuning script for KRONOS model — three-way data source support.

Adds a DistributedThreeWaySampler on top of yctao's version so that
CODEX (primary), IMC (secondary), and CODEX-islet patches (tertiary)
can each be controlled to a fixed per-batch fraction.

All kronos/* modules (including mask_loss, DinoMaskCollator) are imported
from the working directory, which should be yctao's BioSpatialFM root.
Run via: cd /path/to/yctao/BioSpatialFM && torchrun ... /path/to/this/script
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
from torch.utils.data import DataLoader, ConcatDataset
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
    load_marker_metadata,
)
import wandb


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

class DistributedThreeWaySampler(torch.utils.data.Sampler):
    """
    Distributed-aware sampler for three data sources.

    Index layout in ConcatDataset:
      [0, n_primary)                                -> primary (CODEX)
      [n_primary, n_primary+n_secondary)             -> secondary (IMC)
      [n_primary+n_secondary, total)                 -> tertiary (islet patches)

    Epoch length is bounded by the primary shard so every primary patch is
    seen exactly once per epoch.  Secondary and tertiary pools both CYCLE
    with fresh reshuffles to fill their slots in every batch.

    Args:
        n_primary:            total primary samples in the ConcatDataset
        n_secondary:          total secondary samples
        n_tertiary:           total tertiary samples
        batch_size:           per-GPU batch size
        secondary_per_batch:  how many secondary samples per batch
        tertiary_per_batch:   how many tertiary samples per batch
        rank / world_size:    standard DDP identifiers
        seed:                 base seed (mixed with epoch for shuffling)
    """

    def __init__(self, n_primary, n_secondary, n_tertiary,
                 batch_size, secondary_per_batch, tertiary_per_batch,
                 rank, world_size, seed=0):
        primary_per_batch = batch_size - secondary_per_batch - tertiary_per_batch
        assert primary_per_batch > 0, (
            f"secondary_per_batch({secondary_per_batch}) + "
            f"tertiary_per_batch({tertiary_per_batch}) must be < batch_size({batch_size})"
        )
        self.n_primary = n_primary
        self.n_secondary = n_secondary
        self.n_tertiary = n_tertiary
        self.batch_size = batch_size
        self.primary_per_batch = primary_per_batch
        self.secondary_per_batch = secondary_per_batch
        self.tertiary_per_batch = tertiary_per_batch
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

        self.primary_per_rank = math.ceil(n_primary / world_size)
        self.secondary_per_rank = math.ceil(n_secondary / world_size)
        self.tertiary_per_rank = math.ceil(n_tertiary / world_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # ---- primary (CODEX): partition across ranks, no cycling ----
        primary_all = torch.randperm(self.n_primary, generator=g).tolist()
        pad = self.primary_per_rank * self.world_size - len(primary_all)
        primary_all += primary_all[:pad]
        primary_shard = primary_all[
            self.rank * self.primary_per_rank:(self.rank + 1) * self.primary_per_rank
        ]

        # ---- secondary (IMC): partition across ranks, then cycle ----
        secondary_all = (
            torch.randperm(self.n_secondary, generator=g) + self.n_primary
        ).tolist()
        pad = self.secondary_per_rank * self.world_size - len(secondary_all)
        secondary_all += secondary_all[:pad]
        secondary_shard = secondary_all[
            self.rank * self.secondary_per_rank:(self.rank + 1) * self.secondary_per_rank
        ]

        # ---- tertiary (islet): partition across ranks, then cycle ----
        tertiary_all = (
            torch.randperm(self.n_tertiary, generator=g)
            + self.n_primary + self.n_secondary
        ).tolist()
        pad = self.tertiary_per_rank * self.world_size - len(tertiary_all)
        tertiary_all += tertiary_all[:pad]
        tertiary_shard = tertiary_all[
            self.rank * self.tertiary_per_rank:(self.rank + 1) * self.tertiary_per_rank
        ]

        n_batches = len(primary_shard) // self.primary_per_batch

        def _cycle_fill(pool, n_needed, seed_extra):
            g2 = torch.Generator()
            g2.manual_seed(self.seed + self.epoch * 7919 + self.rank + seed_extra)
            result = []
            while len(result) < n_needed:
                order = torch.randperm(len(pool), generator=g2).tolist()
                result.extend(pool[i] for i in order)
            return result[:n_needed]

        secondary_pool = _cycle_fill(secondary_shard, n_batches * self.secondary_per_batch, 1)
        tertiary_pool  = _cycle_fill(tertiary_shard,  n_batches * self.tertiary_per_batch,  2)

        for i in range(n_batches):
            batch = (
                primary_shard  [i * self.primary_per_batch   :(i + 1) * self.primary_per_batch  ] +
                secondary_pool [i * self.secondary_per_batch :(i + 1) * self.secondary_per_batch] +
                tertiary_pool  [i * self.tertiary_per_batch  :(i + 1) * self.tertiary_per_batch ]
            )
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.primary_per_rank // self.primary_per_batch


class DistributedBalancedSampler(torch.utils.data.Sampler):
    """Two-way sampler (primary + secondary). Fallback when islet path is absent."""

    def __init__(self, n_primary, n_secondary, batch_size, secondary_per_batch,
                 rank, world_size, seed=0):
        assert 0 < secondary_per_batch < batch_size
        self.n_primary = n_primary
        self.n_secondary = n_secondary
        self.batch_size = batch_size
        self.secondary_per_batch = int(secondary_per_batch)
        self.primary_per_batch = batch_size - self.secondary_per_batch
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.primary_per_rank = math.ceil(n_primary / world_size)
        self.secondary_per_rank = math.ceil(n_secondary / world_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        primary_all = torch.randperm(self.n_primary, generator=g).tolist()
        primary_all += primary_all[:(self.primary_per_rank * self.world_size - len(primary_all))]
        primary_shard = primary_all[self.rank * self.primary_per_rank:(self.rank + 1) * self.primary_per_rank]

        secondary_all = (torch.randperm(self.n_secondary, generator=g) + self.n_primary).tolist()
        secondary_all += secondary_all[:(self.secondary_per_rank * self.world_size - len(secondary_all))]
        secondary_shard = secondary_all[self.rank * self.secondary_per_rank:(self.rank + 1) * self.secondary_per_rank]

        n_batches = len(primary_shard) // self.primary_per_batch
        secondary_needed = n_batches * self.secondary_per_batch
        g2 = torch.Generator()
        g2.manual_seed(self.seed + self.epoch * 7919 + self.rank)
        secondary_pool = []
        while len(secondary_pool) < secondary_needed:
            order = torch.randperm(len(secondary_shard), generator=g2).tolist()
            secondary_pool.extend(secondary_shard[i] for i in order)
        secondary_pool = secondary_pool[:secondary_needed]

        for i in range(n_batches):
            batch = (
                primary_shard[i * self.primary_per_batch:(i + 1) * self.primary_per_batch] +
                secondary_pool[i * self.secondary_per_batch:(i + 1) * self.secondary_per_batch]
            )
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.primary_per_rank // self.primary_per_batch


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class LossLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / 'loss_curves.csv'
        with open(self.log_file, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'iteration', 'total_loss', 'cls_loss', 'mim_loss', 'lr', 'wd'])

    def log(self, epoch, iteration, total_loss, cls_loss, mim_loss, lr, wd):
        with open(self.log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, iteration, total_loss, cls_loss, mim_loss, lr, wd])

    def save_epoch_summary(self, epoch, train_stats):
        summary_file = self.log_dir / 'epoch_summary.csv'
        file_exists = summary_file.exists()
        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'avg_total_loss', 'avg_cls_loss', 'avg_mim_loss', 'avg_lr', 'avg_wd'])
            writer.writerow([
                epoch,
                train_stats.get('loss', 0), train_stats.get('cls_loss', 0),
                train_stats.get('mim_loss', 0), train_stats.get('lr', 0),
                train_stats.get('wd', 0),
            ])


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser('KRONOS Fine-tuning (three-way)', add_help=False)

    # Model
    parser.add_argument('--model_type', default='vits16', type=str, choices=['vits16', 'vitl16'])
    parser.add_argument('--pretrained_weights', default='hf_hub:MahmoodLab/kronos', type=str)
    parser.add_argument('--token_overlap', action='store_true')
    parser.add_argument('--drop_path_rate', default=0.3, type=float)

    # Training
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.004, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    parser.add_argument('--warmup_epochs', default=10, type=int)
    parser.add_argument('--weight_decay', default=0.04, type=float)
    parser.add_argument('--weight_decay_end', default=0.1, type=float)
    parser.add_argument('--clip_grad', default=3.0, type=float)
    parser.add_argument('--freeze_last_layer', default=1, type=int)

    # DINO
    parser.add_argument('--out_dim', default=65536, type=int)
    parser.add_argument('--norm_last_layer', action='store_true')
    parser.add_argument('--momentum_teacher', default=0.992, type=float)
    parser.add_argument('--use_bn_in_head', action='store_true')

    # Temperature
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float)
    parser.add_argument('--teacher_temp', default=0.07, type=float)
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int)
    parser.add_argument('--student_temp', default=0.1, type=float)
    parser.add_argument('--mim_loss_weight', default=1.0, type=float)
    parser.add_argument('--koleo_loss_weight', default=0.1, type=float)
    parser.add_argument('--grad_accum_steps', default=1, type=int)

    # MIM
    parser.add_argument('--mask_ratio', default=0.5, type=float)
    parser.add_argument('--mask_global_crops_only', action='store_true', default=True)

    # Channel-mask consistency loss (Lmask)
    parser.add_argument('--mask_loss_weight', default=0.0, type=float)
    parser.add_argument('--mask_warmup_epochs', default=1, type=int)
    parser.add_argument('--mask_keep_min', default=0.05, type=float)
    parser.add_argument('--mask_keep_max', default=0.4, type=float)
    parser.add_argument('--mask_min_kept_channels', default=3, type=int)
    parser.add_argument('--mask_n_student_views', default=2, type=int)
    parser.add_argument('--mask_always_keep_markers', default='DAPI', type=str)

    # Augmentation
    parser.add_argument('--global_crops_scale', default=(0.48, 1.0), type=tuple)
    parser.add_argument('--local_crops_scale', default=(0.16, 0.48), type=tuple)
    parser.add_argument('--local_crops_number', default=8, type=int)
    parser.add_argument('--global_crops_size', default=224, type=int)
    parser.add_argument('--local_crops_size', default=96, type=int)

    # ---- Dataset paths ----
    parser.add_argument('--data_path', required=True, type=str, nargs='+',
                        help='Path(s) to primary CODEX patches.')
    parser.add_argument('--data_path_imc', default='', type=str,
                        help='Path to IMC patches (secondary). Flat directory.')
    parser.add_argument('--data_path_islet', default='', type=str,
                        help='Path to CODEX islet patches (tertiary). Nested HPAP-XXX dirs.')
    parser.add_argument('--imc_fraction', default=0.10, type=float,
                        help='Fraction of each batch from IMC. '
                             'secondary_per_batch = round(batch_size * imc_fraction).')
    parser.add_argument('--islet_fraction', default=0.10, type=float,
                        help='Fraction of each batch from islet patches. '
                             'tertiary_per_batch = round(batch_size * islet_fraction).')

    parser.add_argument('--dataset_type', default='patch', type=str,
                        choices=['folder', 'patch', 'list'])
    parser.add_argument('--marker_metadata', default='', type=str,
                        help='Marker metadata CSV for CODEX (primary + islet).')
    parser.add_argument('--marker_metadata_imc', default='', type=str,
                        help='Marker metadata CSV for IMC; merged at startup.')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--bf16', action='store_true',
                        help='Use bfloat16 (recommended on H200/A100).')

    # Checkpoint
    parser.add_argument('--output_dir', default='./output/finetune', type=str)
    parser.add_argument('--saveckp_freq', default=1, type=int)
    parser.add_argument('--resume', default='', type=str)

    # WandB
    parser.add_argument('--wandb_project', default='', type=str)
    parser.add_argument('--wandb_run_name', default='', type=str)

    # Distributed
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://', type=str)

    return parser


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                    data_loader, optimizer, lr_schedule, wd_schedule,
                    momentum_schedule, epoch, fp16_scaler, args, loss_logger=None,
                    use_wandb=False, koleo_loss_fn=None, mask_loss_fn=None):
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}/{args.epochs}]'

    grad_accum_steps = getattr(args, 'grad_accum_steps', 1)
    koleo_weight = getattr(args, 'koleo_loss_weight', 0.0)
    mask_weight_target = getattr(args, 'mask_loss_weight', 0.0)
    mask_warmup_epochs = max(0, getattr(args, 'mask_warmup_epochs', 0))
    use_mask_loss = mask_weight_target > 0.0

    optimizer.zero_grad()

    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if isinstance(batch, dict):
            images = batch['dino_crops']
            marker_ids = batch['dino_marker_ids']
            mask_groups = batch['mask_groups']
        else:
            images, marker_ids = batch
            mask_groups = None

        it_global = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it_global]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it_global]

        if use_mask_loss and mask_warmup_epochs > 0:
            mask_weight = mask_weight_target * min(
                1.0, (epoch + (it / max(1, len(data_loader)))) / mask_warmup_epochs
            )
        else:
            mask_weight = mask_weight_target

        if isinstance(images, list):
            images = [im.cuda(non_blocking=True) for im in images]
            if marker_ids and isinstance(marker_ids[0], list):
                marker_ids = [[m.cuda(non_blocking=True) for m in per_crop]
                              for per_crop in marker_ids]
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
                epoch=epoch,
            )

            koleo_loss_val = torch.tensor(0.0, device=total_loss.device)
            if koleo_loss_fn is not None and koleo_weight > 0:
                backbone_cls = student_ret.get("backbone_cls_tokens")
                if backbone_cls is not None:
                    koleo_loss_val = koleo_loss_fn(backbone_cls)
                    total_loss = total_loss + koleo_weight * koleo_loss_val

            mask_loss_val = torch.tensor(0.0, device=total_loss.device)
            mask_koleo_val = torch.tensor(0.0, device=total_loss.device)
            if use_mask_loss and mask_weight > 0.0 and mask_groups and mask_loss_fn is not None:
                teacher_x = [grp['teacher_imgs'] for grp in mask_groups]
                teacher_m = [grp['teacher_ids'] for grp in mask_groups]
                with torch.no_grad():
                    mt_ret = teacher(teacher_x, marker_ids=teacher_m, mask_branch=True)
                    z_t = mt_ret["x_norm_clstoken"].detach()

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

                if student_backbone_cls_list:
                    mask_backbone_cls = torch.cat(student_backbone_cls_list, dim=0)
                    mask_koleo_val = koleo_loss_fn(mask_backbone_cls)
                    total_loss = total_loss + koleo_weight * mask_koleo_val

        if not torch.isfinite(total_loss):
            print(f"Loss is {total_loss}, stopping training")
            sys.exit(1)

        scaled_loss = total_loss / grad_accum_steps
        if fp16_scaler is None:
            scaled_loss.backward()
        else:
            fp16_scaler.scale(scaled_loss).backward()

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

            with torch.no_grad():
                m = momentum_schedule[it_global]
                for param_q, param_k in zip(student.parameters(), teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

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

        if loss_logger is not None:
            loss_logger.log(
                epoch=epoch, iteration=it_global,
                total_loss=total_loss.item(), cls_loss=cls_loss.item(),
                mim_loss=mim_loss.item(),
                lr=optimizer.param_groups[0]["lr"],
                wd=optimizer.param_groups[0]["weight_decay"],
            )

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
            mask_str = (f", Mask: {mask_loss_val.item():.4f}@w={mask_weight:.3f}"
                        if use_mask_loss else "")
            print(f"  [Iter {it:4d}] Loss: {total_loss.item():.4f} "
                  f"(CLS: {cls_loss.item():.4f}, MIM: {mim_loss.item():.4f}, "
                  f"KoLeo: {koleo_loss_val.item():.4f}{mask_str}) | "
                  f"LR: {optimizer.param_groups[0]['lr']:.8f} | "
                  f"WD: {optimizer.param_groups[0]['weight_decay']:.6f}")

    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clip_gradients(model, clip):
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)


def cancel_gradients_last_layer(epoch, student, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in student.named_parameters():
        if "last_layer" in n:
            p.grad = None


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)

    def __str__(self):
        return self.delimiter.join(
            f"{name}: {meter.avg:.8f}" if name == 'lr' else f"{name}: {meter.avg:.4f}"
            for name, meter in self.meters.items()
        )

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        header = header or ''
        end = time.time()
        iter_time = AverageMeter()
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time:.4f}',
        ])
        for obj in iterable:
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_string = str(int(iter_time.avg * (len(iterable) - i)))
                print(log_msg.format(i, len(iterable), eta=eta_string,
                                     meters=str(self), time=iter_time.avg))
            i += 1
            end = time.time()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

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
        self.sum, self.count = t[0], t[1]
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(checkpoint_path, student, teacher, optimizer, fp16_scaler,
                    dino_loss=None, mask_loss_fn=None):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    student.load_state_dict(checkpoint['student'])
    teacher.load_state_dict(checkpoint['teacher'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if fp16_scaler is not None and 'fp16_scaler' in checkpoint:
        fp16_scaler.load_state_dict(checkpoint['fp16_scaler'])
    if dino_loss is not None and 'dino_loss' in checkpoint:
        dino_loss.load_state_dict(checkpoint['dino_loss'])
    if (mask_loss_fn is not None and 'mask_loss' in checkpoint
            and checkpoint['mask_loss'] is not None):
        mask_loss_fn.load_state_dict(checkpoint['mask_loss'])
        print(f"  mask_loss center buffer restored "
              f"(norm={mask_loss_fn.center.norm().item():.4f})")
    return checkpoint['epoch']


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # ---- Distributed setup ----
    if args.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        rank = int(os.environ.get("RANK", local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
        args.local_rank = local_rank
        args.world_size = world_size
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=rank)
        print(f"Distributed: rank {rank}/{world_size} on GPU {local_rank}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ---- WandB ----
    use_wandb = WANDB_AVAILABLE and args.wandb_project and args.local_rank == 0
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name or None,
                   config=vars(args), dir=args.output_dir)
        print(f"WandB: project={args.wandb_project}, run={wandb.run.name}")
    elif args.wandb_project and not WANDB_AVAILABLE:
        print("Warning: --wandb_project set but wandb not installed.")

    loss_logger = None
    if args.local_rank == 0:
        log_dir = os.path.join(args.output_dir, 'logs')
        loss_logger = LossLogger(log_dir)
        print(f"Loss curves → {log_dir}")
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    # ---- Marker metadata ----
    if not args.marker_metadata:
        raise ValueError("--marker_metadata must be provided.")
    marker_metadata = load_marker_metadata(args.marker_metadata)
    if args.marker_metadata_imc:
        marker_metadata_imc = load_marker_metadata(args.marker_metadata_imc)
        overlap = set(marker_metadata) & set(marker_metadata_imc)
        if overlap:
            print(f"Warning: {len(overlap)} overlapping markers: {overlap}")
        marker_metadata = {**marker_metadata, **marker_metadata_imc}
        print(f"Merged marker metadata: {len(marker_metadata)} markers total")

    # ---- Transform ----
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

    # ---- Datasets ----
    def _make_dataset(data_path, recursive=False):
        if args.dataset_type == 'folder':
            return MultiplexImageFolderDataset(data_root=data_path, transform=transform)
        elif args.dataset_type == 'patch':
            return MultiplexPatchDataset(patch_dir=data_path, transform=transform,
                                         recursive=recursive)
        else:
            raise ValueError(f"Unknown dataset_type: {args.dataset_type}")

    # Primary: CODEX (recursive handles both flat and nested layouts)
    primary_paths = args.data_path if isinstance(args.data_path, list) else [args.data_path]
    primary_datasets = [_make_dataset(p, recursive=True) for p in primary_paths]
    for p, d in zip(primary_paths, primary_datasets):
        print(f"  primary  {p}: {len(d)} patches")
    dataset = ConcatDataset(primary_datasets) if len(primary_datasets) > 1 else primary_datasets[0]
    n_primary = len(dataset)

    # Secondary: IMC (flat directory)
    n_imc = 0
    if args.data_path_imc:
        dataset_imc = _make_dataset(args.data_path_imc, recursive=False)
        n_imc = len(dataset_imc)
        print(f"  secondary (IMC)   {args.data_path_imc}: {n_imc} patches")
        dataset = ConcatDataset([dataset, dataset_imc])

    # Tertiary: CODEX islet patches (nested HPAP-XXX subdirs, same marker space as primary)
    n_islet = 0
    if args.data_path_islet:
        dataset_islet = _make_dataset(args.data_path_islet, recursive=True)
        n_islet = len(dataset_islet)
        print(f"  tertiary (islet)  {args.data_path_islet}: {n_islet} patches")
        dataset = ConcatDataset([dataset, dataset_islet])

    print(f"Total: primary={n_primary}, imc={n_imc}, islet={n_islet}, "
          f"sum={n_primary + n_imc + n_islet}")

    # ---- Collator ----
    if use_mask_loss:
        from kronos.dataset import DinoMaskCollator
        always_keep_marker_ids = []
        for name in [n.strip() for n in (args.mask_always_keep_markers or '').split(',') if n.strip()]:
            meta = marker_metadata.get(name)
            if meta is None:
                print(f"  [Lmask] Warning: '{name}' not in marker_metadata; ignoring.")
            else:
                mid = int(meta.get('marker_id', 0))
                always_keep_marker_ids.append(mid)
                print(f"  [Lmask] always-keep '{name}' (marker_id={mid})")
        collate_fn = DinoMaskCollator(
            keep_min=args.mask_keep_min,
            keep_max=args.mask_keep_max,
            always_keep_marker_ids=always_keep_marker_ids,
            n_student_views=args.mask_n_student_views,
            min_kept_channels=args.mask_min_kept_channels,
        )
    else:
        collate_fn = collate_fn_multicrop

    # ---- Sampler / DataLoader ----
    if args.distributed:
        rank = int(os.environ.get('RANK', args.local_rank))
        ws = int(os.environ.get('WORLD_SIZE', args.world_size))

        has_imc   = bool(args.data_path_imc)   and n_imc   > 0 and args.imc_fraction   > 0
        has_islet = bool(args.data_path_islet) and n_islet > 0 and args.islet_fraction > 0

        if has_imc and has_islet:
            secondary_per_batch = max(1, int(round(args.batch_size * args.imc_fraction)))
            tertiary_per_batch  = max(1, int(round(args.batch_size * args.islet_fraction)))
            assert secondary_per_batch + tertiary_per_batch < args.batch_size, (
                "imc_fraction + islet_fraction leaves no room for CODEX samples"
            )
            sampler = DistributedThreeWaySampler(
                n_primary=n_primary, n_secondary=n_imc, n_tertiary=n_islet,
                batch_size=args.batch_size,
                secondary_per_batch=secondary_per_batch,
                tertiary_per_batch=tertiary_per_batch,
                rank=rank, world_size=ws,
            )
            print(f"DistributedThreeWaySampler: {len(sampler)} batches/rank/epoch | "
                  f"per batch: {sampler.primary_per_batch} CODEX + "
                  f"{sampler.secondary_per_batch} IMC + "
                  f"{sampler.tertiary_per_batch} islet | "
                  f"IMC cycles ~{(len(sampler)*secondary_per_batch) / max(1, sampler.secondary_per_rank):.2f}x/epoch | "
                  f"islet cycles ~{(len(sampler)*tertiary_per_batch) / max(1, sampler.tertiary_per_rank):.2f}x/epoch")
            data_loader = DataLoader(
                dataset, batch_sampler=sampler,
                num_workers=args.num_workers, pin_memory=True,
                collate_fn=collate_fn,
            )

        elif has_imc:
            secondary_per_batch = max(1, int(round(args.batch_size * args.imc_fraction)))
            sampler = DistributedBalancedSampler(
                n_primary=n_primary, n_secondary=n_imc,
                batch_size=args.batch_size,
                secondary_per_batch=secondary_per_batch,
                rank=rank, world_size=ws,
            )
            print(f"DistributedBalancedSampler (CODEX+IMC): {len(sampler)} batches/rank/epoch | "
                  f"{sampler.primary_per_batch} CODEX + {sampler.secondary_per_batch} IMC per batch")
            data_loader = DataLoader(
                dataset, batch_sampler=sampler,
                num_workers=args.num_workers, pin_memory=True,
                collate_fn=collate_fn,
            )

        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            data_loader = DataLoader(
                dataset, sampler=sampler,
                batch_size=args.batch_size,
                num_workers=args.num_workers, pin_memory=True,
                drop_last=True, collate_fn=collate_fn,
            )
    else:
        sampler = None
        data_loader = DataLoader(
            dataset, shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True, drop_last=True,
            collate_fn=collate_fn,
        )

    # ---- Models ----
    print("Creating student model...")
    student_backbone, precision, embed_dim = create_model_from_pretrained(
        checkpoint_path=args.pretrained_weights,
        cfg={"model_type": args.model_type, "token_overlap": args.token_overlap,
             "drop_path_rate": args.drop_path_rate},
    )
    print("Creating teacher model...")
    teacher_backbone, _, _ = create_model_from_pretrained(
        checkpoint_path=args.pretrained_weights,
        cfg={"model_type": args.model_type, "token_overlap": args.token_overlap,
             "drop_path_rate": args.drop_path_rate},
    )

    student_head = DINOHead(in_dim=embed_dim, out_dim=args.out_dim,
                            use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer)
    teacher_head = DINOHead(in_dim=embed_dim, out_dim=args.out_dim,
                            use_bn=args.use_bn_in_head)

    print(f"MIM: mask_ratio={args.mask_ratio}, global_crops_only={args.mask_global_crops_only}")
    student = MultiCropWrapper(student_backbone, student_head,
                               mask_ratio=args.mask_ratio,
                               mask_global_crops_only=args.mask_global_crops_only)
    teacher = MultiCropWrapper(teacher_backbone, teacher_head,
                               mask_ratio=0.0, mask_global_crops_only=True)

    student = student.cuda()
    teacher = teacher.cuda()
    teacher.load_state_dict(student.state_dict(), strict=False)
    for p in teacher.parameters():
        p.requires_grad = False

    if args.distributed:
        student = DDP(student, device_ids=[args.local_rank], find_unused_parameters=True)
    teacher_without_ddp = teacher
    print(f"Student/Teacher ready. embed_dim={embed_dim}")

    # ---- Losses ----
    koleo_loss_fn = None
    if getattr(args, 'koleo_loss_weight', 0.0) > 0:
        koleo_loss_fn = KoLeoLoss().cuda()
        print(f"KoLeo: weight={args.koleo_loss_weight}")

    mask_loss_fn = None
    if use_mask_loss:
        from kronos.mask_loss import MaskCEloss
        mask_loss_fn = MaskCEloss(
            out_dim=args.out_dim,
            teacher_temp=args.teacher_temp,
            student_temp=args.student_temp,
            center_momentum=0.9,
        ).cuda()
        print(f"Mask CE loss: weight={args.mask_loss_weight}, "
              f"warmup={args.mask_warmup_epochs}, n_views={args.mask_n_student_views}, "
              f"keep~U[{args.mask_keep_min},{args.mask_keep_max}], "
              f"min_kept={args.mask_min_kept_channels}")

    if getattr(args, 'grad_accum_steps', 1) > 1:
        print(f"Grad accum: {args.grad_accum_steps} steps "
              f"(eff. batch = {args.batch_size * args.grad_accum_steps})")

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

    # ---- Optimizer & schedules ----
    optimizer = torch.optim.AdamW(
        [{'params': [p for p in student.parameters() if p.requires_grad]}],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    effective_batch = args.batch_size * args.world_size
    base_lr = args.lr if effective_batch < 32 else args.lr * effective_batch / 128.
    print(f"Effective batch {effective_batch}, base_lr={base_lr:.6f}")

    lr_schedule = cosine_scheduler(base_lr, args.min_lr, args.epochs, len(data_loader),
                                   warmup_epochs=args.warmup_epochs, start_warmup_value=0)
    wd_schedule = cosine_scheduler(args.weight_decay, args.weight_decay_end,
                                   args.epochs, len(data_loader))
    momentum_schedule = cosine_scheduler(args.momentum_teacher, 1,
                                         args.epochs, len(data_loader))

    fp16_scaler = None
    if args.bf16:
        precision = torch.bfloat16
    elif precision == torch.float16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ---- Resume ----
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(
            args.resume, student, teacher, optimizer, fp16_scaler,
            dino_loss=dino_loss, mask_loss_fn=mask_loss_fn,
        ) + 1
        print(f"Resumed: starting from epoch {start_epoch}")

    # ---- Training loop ----
    print(f"Starting training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            bs = data_loader.batch_sampler
            if hasattr(bs, 'set_epoch'):
                bs.set_epoch(epoch)
            elif hasattr(data_loader.sampler, 'set_epoch'):
                data_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule,
            momentum_schedule, epoch, fp16_scaler, args, loss_logger,
            use_wandb=use_wandb, koleo_loss_fn=koleo_loss_fn,
            mask_loss_fn=mask_loss_fn,
        )

        if args.local_rank == 0 and loss_logger is not None:
            loss_logger.save_epoch_summary(epoch, train_stats)

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

        if args.local_rank == 0:
            if (epoch + 1) % args.saveckp_freq == 0 or epoch == args.epochs - 1:
                save_dict = {
                    'student': student.state_dict(),
                    'teacher': teacher.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'dino_loss': dino_loss.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'mask_loss': mask_loss_fn.state_dict() if mask_loss_fn is not None else None,
                }
                if fp16_scaler is not None:
                    save_dict['fp16_scaler'] = fp16_scaler.state_dict()
                save_checkpoint(save_dict,
                                os.path.join(args.output_dir, f'checkpoint_{epoch:04d}.pth'))
                save_checkpoint(save_dict,
                                os.path.join(args.output_dir, 'checkpoint_latest.pth'))

    print(f'Training time: {(time.time() - start_time) / 3600:.2f} hours')
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('KRONOS Fine-tuning (three-way)',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
