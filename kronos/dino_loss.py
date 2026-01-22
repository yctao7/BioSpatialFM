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
    DINO loss implementation for self-supervised learning with MIM support.
    
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
        self.register_buffer("center_patches", torch.zeros(1, out_dim))
        
        # Teacher temperature schedule
        self.teacher_temp_schedule = torch.cat((
            torch.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            torch.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        
    def forward(self, student_output, teacher_output, student_patch_out, teacher_patch_out, masks, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        
        Args:
            student_output: Output from student network (all crops) - [B*ncrops, out_dim]
            teacher_output: Output from teacher network (only global crops) - [B*2, out_dim]
            student_patch_out: Dict of patch tokens from student - {num_patches: [B*count, num_patches, out_dim]}
            teacher_patch_out: Dict of patch tokens from teacher - {num_patches: [B*2, num_patches, out_dim]}
            masks: Dict of masks - {num_patches: [B*count, num_patches]}
            epoch: Current epoch for temperature scheduling
        """
        # CLS token loss
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

        # Patch-level MIM loss (only for global crops to save memory)
        patch_loss = torch.tensor(0.0, device=teacher_output.device)
        if (student_patch_out is not None and teacher_patch_out is not None and 
            isinstance(student_patch_out, dict) and isinstance(teacher_patch_out, dict)):
            
            # Only compute patch loss for resolutions that exist in both teacher and student
            for num_patches in teacher_patch_out.keys():
                if num_patches not in student_patch_out:
                    continue
                
                t_patches = teacher_patch_out[num_patches]  # [B*2, num_patches, out_dim]
                s_patches = student_patch_out[num_patches]  # [B*count, num_patches, out_dim]
                
                B_teacher, n_patches, out_dim = t_patches.shape
                
                # For student, only use the first B_teacher samples (matching teacher's global crops)
                s_patches_global = s_patches[:B_teacher]  # [B*2, num_patches, out_dim]
                
                # Process in chunks to save memory
                chunk_size = 512  # Process 512 patches at a time
                chunk_losses = []
                chunk_weights = []
                
                for i in range(0, n_patches, chunk_size):
                    end_idx = min(i + chunk_size, n_patches)
                    
                    # Get patch chunks
                    t_chunk = t_patches[:, i:end_idx, :].reshape(-1, out_dim)  # [B*2*chunk, out_dim]
                    s_chunk = s_patches_global[:, i:end_idx, :].reshape(-1, out_dim)
                    
                    # Apply softmax with centering for teacher
                    t_patch = F.softmax((t_chunk - self.center_patches) / temp, dim=-1)
                    t_patch = t_patch.detach()

                    # Apply log_softmax for student
                    s_patch = F.log_softmax(s_chunk / self.student_temp, dim=-1)

                    # Compute loss
                    loss_mim = torch.sum(-t_patch * s_patch, dim=-1)  # [B*2*chunk]
                    
                    # Apply masks if available
                    if masks is not None and num_patches in masks:
                        mask_chunk = masks[num_patches][:B_teacher, i:end_idx].reshape(-1)
                        chunk_losses.append((loss_mim * mask_chunk).sum())
                        chunk_weights.append(mask_chunk.sum())
                    else:
                        chunk_losses.append(loss_mim.sum())
                        chunk_weights.append(torch.tensor(loss_mim.numel(), device=loss_mim.device))
                
                # Aggregate across chunks
                if chunk_losses:
                    total_loss_tensor = torch.stack(chunk_losses).sum()
                    total_weight_tensor = torch.stack(chunk_weights).sum()
                    patch_loss = patch_loss + total_loss_tensor / total_weight_tensor.clamp(min=1.0)

        self.update_center(teacher_output, teacher_patch_out)
        
        return total_loss + patch_loss

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

        if teacher_patch_out is not None and isinstance(teacher_patch_out, dict):
            # Collect all patches from all resolutions
            all_patches = []
            for patches in teacher_patch_out.values():
                B, num_patches, out_dim = patches.shape
                all_patches.append(patches.reshape(-1, out_dim))
            
            if all_patches:
                all_patches_cat = torch.cat(all_patches, dim=0)
                batch_center_p = torch.sum(all_patches_cat, dim=0, keepdim=True)

                if dist.is_initialized():
                    dist.all_reduce(batch_center_p)
                    batch_center_p = batch_center_p / dist.get_world_size()

                batch_center_p = batch_center_p / len(all_patches_cat)

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
        cls_outputs = []
        patch_outputs_by_size = {}  # Group by number of patches
        masks_by_size = {}
        
        for size, count in zip(unique_sizes, counts):
            end_idx = start_idx + count
            
            # Concatenate crops of the same resolution
            _out = torch.cat(x[start_idx:end_idx])
            
            # Get marker_ids for this batch
            if marker_ids is not None:
                batch_size = x[start_idx].shape[0]
                _marker_ids = []
                for i in range(start_idx, end_idx):
                    # Each crop has batch_size samples
                    _marker_ids.extend([marker_ids[i]] * batch_size)
            else:
                _marker_ids = None
            
            # Forward through backbone
            backbone_out = self.backbone(_out, marker_ids=_marker_ids, is_training=True)
            
            # Collect CLS token outputs
            cls_outputs.append(backbone_out["x_norm_clstoken"])
            
            # Collect patch tokens grouped by number of patches
            if "x_norm_patchtokens" in backbone_out and backbone_out["x_norm_patchtokens"] is not None:
                patches = backbone_out["x_norm_patchtokens"]
                num_patches = patches.shape[1]
                if num_patches not in patch_outputs_by_size:
                    patch_outputs_by_size[num_patches] = []
                patch_outputs_by_size[num_patches].append(patches)
            
            # Collect masks grouped by number of patches
            if "masks" in backbone_out and backbone_out["masks"] is not None:
                mask = backbone_out["masks"]
                num_patches = mask.shape[1] if mask.dim() > 1 else mask.shape[0]
                if num_patches not in masks_by_size:
                    masks_by_size[num_patches] = []
                masks_by_size[num_patches].append(mask)
            
            start_idx = end_idx
        
        # Concatenate all CLS token outputs and pass through head
        cls_concat = torch.cat(cls_outputs)
        head_output = self.head(cls_concat)
        
        # Prepare return dictionary
        ret = {
            "x_norm_clstoken": head_output,
        }
        
        # Process patch tokens grouped by size
        if patch_outputs_by_size:
            patch_dict = {}
            for num_patches, patch_list in patch_outputs_by_size.items():
                patch_concat = torch.cat(patch_list, dim=0)  # [B_total, num_patches, embed_dim]
                B, n_p, embed_dim = patch_concat.shape
                patch_flat = patch_concat.reshape(-1, embed_dim)
                patch_head_out = self.head(patch_flat)
                patch_dict[num_patches] = patch_head_out.reshape(B, n_p, -1)
            ret["x_norm_patchtokens"] = patch_dict
        
        # Process masks grouped by size
        if masks_by_size:
            mask_dict = {}
            for num_patches, mask_list in masks_by_size.items():
                mask_dict[num_patches] = torch.cat(mask_list, dim=0)
            ret["masks"] = mask_dict
        
        return ret