"""
Fine-tuning script for KRONOS model using DINO self-supervised learning.
Now with proper MIM (Masked Image Modeling) support.
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

import torch    
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import csv
from kronos import create_model_from_pretrained
from kronos.dino_head import DINOHead
from kronos.dino_loss import DINOLoss, MultiCropWrapper
from kronos.data_augmentation import get_augmentation_pipeline
from kronos.dataset import (
    MultiplexImageDataset,
    MultiplexImageFolderDataset,
    MultiplexPatchDataset,
    collate_fn_multicrop,
    load_marker_metadata
)


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
    parser.add_argument('--weight_decay_end', default=0.4, type=float,
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
    
    # MIM parameters (NEW)
    parser.add_argument('--mask_ratio', default=0.4, type=float,
                        help='Ratio of patches to mask for MIM (0.0 to disable MIM)')
    parser.add_argument('--mask_global_crops_only', action='store_true', default=True,
                        help='Only apply masking to global crops (recommended)')
    
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
    parser.add_argument('--data_path', required=True, type=str,
                        help='Path to dataset')
    parser.add_argument('--dataset_type', default='patch', type=str,
                        choices=['folder', 'patch', 'list'],
                        help='Type of dataset')
    parser.add_argument('--marker_metadata', default='tutorials/codex_dataset/dataset/marker_info_with_metadata.csv', type=str,
                        help='Path to marker metadata CSV file')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='Number of data loading workers')
    
    # Checkpoint parameters
    parser.add_argument('--output_dir', default='./output/finetune', type=str,
                        help='Path to save checkpoints')
    parser.add_argument('--saveckp_freq', default=20, type=int,
                        help='Save checkpoint every x epochs')
    parser.add_argument('--resume', default='', type=str,
                        help='Path to checkpoint to resume from')
    
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
                   momentum_schedule, epoch, fp16_scaler, args, loss_logger=None):
    """Train for one epoch."""
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}/{args.epochs}]'
    
    for it, (images, marker_ids) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # Update learning rate and weight decay according to schedule
        it_global = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it_global]
            if i == 0:  # Only first group has weight decay
                param_group["weight_decay"] = wd_schedule[it_global]
        
        # Move images to GPU
        if isinstance(images, list):
            images = [im.cuda(non_blocking=True) for im in images]
            marker_ids = [m.cuda(non_blocking=True) for m in marker_ids]
        else:
            images = images.cuda(non_blocking=True)
            marker_ids = marker_ids.cuda(non_blocking=True)
        
        # Forward pass
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # Teacher forward - only global views (first 2 crops), no masking
            teacher_ret = teacher(images[:2], marker_ids=marker_ids[:2], is_student=False)
            
            # Student forward - all views, with masking for MIM
            student_ret = student(images, marker_ids=marker_ids, is_student=True)
            
            # Compute loss (now returns total_loss, cls_loss, mim_loss)
            total_loss, cls_loss, mim_loss = dino_loss(
                student_output=student_ret["x_norm_clstoken"],           # [B*ncrops, out_dim]
                teacher_output=teacher_ret["x_norm_clstoken"],           # [B*2, out_dim]
                student_patch_out=student_ret.get("x_norm_patchtokens"), # {num_patches: [B*count, num_patches, out_dim]}
                teacher_patch_out=teacher_ret.get("x_norm_patchtokens"), # {num_patches: [B*2, num_patches, out_dim]}
                masks=student_ret.get("masks"),                          # {num_patches: [B*count, num_patches]}
                epoch=epoch
            )
            loss = total_loss
        
        if not torch.isfinite(loss):
            print(f"Loss is {loss}, stopping training")
            sys.exit(1)
        
        # Backward pass
        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = clip_gradients(student, args.clip_grad)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = clip_gradients(student, args.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        
        # EMA update for teacher
        with torch.no_grad():
            m = momentum_schedule[it_global]
            for param_q, param_k in zip(student.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        
        # Logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(cls_loss=cls_loss.item())
        metric_logger.update(mim_loss=mim_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        # Log to CSV
        if loss_logger is not None:
            loss_logger.log(
                epoch=epoch,
                iteration=it_global,
                total_loss=loss.item(),
                cls_loss=cls_loss.item(),
                mim_loss=mim_loss.item(),
                lr=optimizer.param_groups[0]["lr"],
                wd=optimizer.param_groups[0]["weight_decay"]
            )

        # logging per 50 iteration
        if it % 50 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            current_wd = optimizer.param_groups[0]["weight_decay"]
            print(f"  [Iter {it:4d}] Loss: {loss.item():.4f} (CLS: {cls_loss.item():.4f}, MIM: {mim_loss.item():.4f}) | LR: {current_lr:.8f} | WD: {current_wd:.6f}")
    
    # Gather stats from all processes
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


def load_checkpoint(checkpoint_path, student, teacher, optimizer, fp16_scaler):
    """Load checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    student.load_state_dict(checkpoint['student'])
    teacher.load_state_dict(checkpoint['teacher'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if fp16_scaler is not None and 'fp16_scaler' in checkpoint:
        fp16_scaler.load_state_dict(checkpoint['fp16_scaler'])
    
    return checkpoint['epoch']


def main(args):
    # Setup distributed training
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                               world_size=args.world_size, rank=args.local_rank)
        print(f"Distributed training enabled: rank {args.local_rank}/{args.world_size}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

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
    
    # Load marker metadata if provided
    if args.marker_metadata:
        marker_metadata = load_marker_metadata(args.marker_metadata)
    else:
        raise ValueError("Marker metadata file must be provided.")
    
    # Setup data augmentation
    transform = get_augmentation_pipeline(
        marker_metadata,
        augmentation_type='dino',
        global_crops_size=args.global_crops_size,
        local_crops_size=args.local_crops_size,
        local_crops_number=args.local_crops_number,
        global_crops_scale=args.global_crops_scale,
        local_crops_scale=args.local_crops_scale,
    )
    
    # Create dataset
    if args.dataset_type == 'folder':
        dataset = MultiplexImageFolderDataset(
            data_root=args.data_path,
            transform=transform,
        )
    elif args.dataset_type == 'patch':
        dataset = MultiplexPatchDataset(
            patch_dir=args.data_path,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_multicrop,
    )
    
    # Create student and teacher models
    print("Creating student model...")
    student_backbone, precision, embed_dim = create_model_from_pretrained(
        checkpoint_path=args.pretrained_weights,
        cfg={"model_type": args.model_type, "token_overlap": args.token_overlap}
    )
    
    print("Creating teacher model...")
    teacher_backbone, _, _ = create_model_from_pretrained(
        checkpoint_path=args.pretrained_weights,
        cfg={"model_type": args.model_type, "token_overlap": args.token_overlap}
    )
    
    # Create DINO heads
    student_head = DINOHead(
        in_dim=embed_dim,
        out_dim=args.out_dim,
        use_bn=args.use_bn_in_head,
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
    
    # Wrap with DDP
    if args.distributed:
        student = DDP(student, device_ids=[args.local_rank])
        teacher = DDP(teacher, device_ids=[args.local_rank])
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher
    
    print(f"Student and Teacher are ready. Embedding dim: {embed_dim}")
    
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
        base_lr = args.lr * effective_batch / 256.
        print(f"Large batch ({effective_batch}), scaling LR: {base_lr}")
    
    # Setup learning rate schedule
    lr_schedule = cosine_scheduler(
        base_lr,
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
        start_warmup_value = 0.004
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
    
    # Setup mixed precision training
    fp16_scaler = None
    if precision == torch.float16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, student, teacher, optimizer, fp16_scaler)
        start_epoch += 1
    
    print(f"Starting training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            data_loader.sampler.set_epoch(epoch)
        
        # Train one epoch
        train_stats = train_one_epoch(
            student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule,
            momentum_schedule, epoch, fp16_scaler, args, loss_logger
        )

        # Save epoch summary
        if args.local_rank == 0 and loss_logger is not None:
            loss_logger.save_epoch_summary(epoch, train_stats)

        # Save checkpoint
        if args.local_rank == 0:
            if (epoch + 1) % args.saveckp_freq == 0 or epoch == args.epochs - 1:
                save_dict = {
                    'student': student.state_dict(),
                    'teacher': teacher.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if fp16_scaler is not None:
                    save_dict['fp16_scaler'] = fp16_scaler.state_dict()
                
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('KRONOS Fine-tuning', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)