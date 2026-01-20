# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class DINOLoss(nn.Module):
    """
    DINO loss implementation for self-supervised learning.
    
    Args:
        out_dim (int): Output dimension of the DINO head
        ncrops (int): Total number of crops (local + global)
        warmup_teacher_temp (float): Initial teacher temperature
        teacher_temp (float): Final teacher temperature
        warmup_teacher_temp_epochs (int): Number of epochs for teacher temperature warmup
        nepochs (int): Total number of training epochs
        student_temp (float): Student temperature
        center_momentum (float): Momentum for updating center
    """
    def __init__(
        self,
        out_dim,
        ncrops=10,
        warmup_teacher_temp=0.04,
        teacher_temp=0.04,
        warmup_teacher_temp_epochs=30,
        nepochs=100,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center_patches", torch.zeros(1, out_dim)) # Add: patch level centered vector
        
        # Teacher temperature schedule
        self.teacher_temp_schedule = torch.cat((
            torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            torch.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        
    def forward(self, student_output, teacher_output, student_patch_out, teacher_patch_out, masks, epoch): # Add: parameters
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        
        Args:
            student_output: Output from student network (all crops)
            teacher_output: Output from teacher network (only global crops)
            epoch: Current epoch for temperature scheduling
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  # Only 2 global views

        total_loss = 0
        n_loss_terms = 0
        
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Skip when student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
                
        total_loss /= n_loss_terms

        patch_loss = 0 #Add : patch loss
        if student_patch_out is not None and teacher_patch_out is not None and masks is not None:
            t_patch = F.softmax((teacher_patch_out - self.center_patches) / temp, dim=-1)
            t_patch = t_patch.detach()

            s_patch = F.log_softmax(student_patch_out / self.student_temp, dim=-1)

            loss_mism = torch.sum(-t_patch * s_patch, dim=-1)
            mask_val = masks.flatten()
            patch_loss = torch.sum(loss_mism * mask_val) / mask_val.sum().clamp(min=1.0)

        self.update_center(teacher_output, teacher_patch_out)
        
        return total_loss + patch_loss #Add: patch loss

    @torch.no_grad()
    def update_center(self, teacher_output, teacher_patch_out=None):
        """
        Update center used for teacher output with exponential moving average.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        
        # Gather from all GPUs if using DDP
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / dist.get_world_size()
        
        batch_center = batch_center / len(teacher_output)
        
        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        if teacher_patch_out is not None:
            batch_center_p = torch.sum(teacher_patch_out, dim=0, keepdim=True)

            if dist.is_initialized():
                dist.all_reduce(batch_center_p)
                batch_center_p = batch_center_p / dist.get_world_size()

            batch_center_p = batch_center_p / len(teacher_patch_out)

            self.center_patches = self.center_patches * self.center_momentum + batch_center_p * (1 - self.center_momentum)


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # Disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x, marker_ids=None):
        # Convert to list if not already
        if not isinstance(x, list):
            x = [x]
        
        # Get unique image sizes and their counts
        sizes = torch.tensor([inp.shape[-1] for inp in x])
        unique_sizes, counts = torch.unique_consecutive(sizes, return_counts=True)
        
        start_idx = 0
        output = []
        
        for size, count in zip(unique_sizes, counts):
            end_idx = start_idx + count
            
            # Concatenate crops of the same resolution
            _out = torch.cat(x[start_idx:end_idx])
            
            # Get marker_ids for this batch
            if marker_ids is not None:
                # When we concatenate crops, we need to create a single list with 
                # marker_ids repeated for the entire concatenated batch
                # Each crop in x[start_idx:end_idx] has batch_size samples
                # After concatenation, _out has batch_size * count samples
                # We need to create a list where marker_ids[i] is repeated batch_size times
                batch_size = x[start_idx].shape[0]
                _marker_ids = []
                for i in range(start_idx, end_idx):
                    # Repeat the marker_ids for each sample in the batch
                    for _ in range(batch_size):
                        _marker_ids.append(marker_ids[i])
            else:
                _marker_ids = None
            
            # Forward through backbone
            _out = self.backbone(_out, marker_ids=_marker_ids, is_training=True)
            output.append(_out["x_norm_clstoken"])
            start_idx = end_idx
            
        # Concatenate all outputs and pass through head
        output = torch.cat(output)
        return self.head(output)
