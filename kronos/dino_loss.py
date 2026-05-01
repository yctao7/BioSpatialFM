# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math


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
        mim_loss_weight (float): Weight for MIM loss
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
        mim_loss_weight=1.0,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.mim_loss_weight = mim_loss_weight
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
            masks: Dict of masks - {num_patches: [B*count, num_patches]} (1 = masked, 0 = visible)
            epoch: Current epoch for temperature scheduling
        """
        # CLS token loss
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # Teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  # Only 2 global views

        cls_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # Skip when student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                cls_loss += loss.mean()
                n_loss_terms += 1

        cls_loss /= n_loss_terms

        # Patch-level MIM loss (only for global crops)
        mim_loss = torch.tensor(0.0, device=teacher_output.device)
        
        if (student_patch_out is not None and teacher_patch_out is not None and
            masks is not None and
            isinstance(student_patch_out, dict) and isinstance(teacher_patch_out, dict) and
            isinstance(masks, dict)):

            total_patch_loss = 0.0
            total_masked_count = 0

            for num_patches in teacher_patch_out.keys():
                if num_patches not in student_patch_out or num_patches not in masks:
                    continue

                t_patches = teacher_patch_out[num_patches]  # [B*2, num_patches, out_dim]
                s_patches = student_patch_out[num_patches]  # [B*count, num_patches, out_dim]
                mask = masks[num_patches]  # [B_mask, num_patches], 1 = masked

                B_teacher, n_patches_t, out_dim = t_patches.shape
                B_student, n_patches_s, _ = s_patches.shape
                B_mask, n_patches_m = mask.shape

                # Verify dimensions match
                if n_patches_t != n_patches_s:
                    print(f"Warning: patch count mismatch t={n_patches_t}, s={n_patches_s}, skipping")
                    continue
                
                n_patches = n_patches_t

                # For student, only use the first B_teacher samples (matching teacher's global crops)
                s_patches_global = s_patches[:B_teacher]  # [B*2, num_patches, out_dim]
                
                # For mask, also only use first B_teacher samples
                # mask might have more samples if it was generated for all crops
                mask_global = mask[:B_teacher]  # [B*2, num_patches]

                # Verify mask dimensions
                if mask_global.shape[0] != B_teacher or mask_global.shape[1] != n_patches:
                    print(f"Warning: mask shape {mask_global.shape} doesn't match expected ({B_teacher}, {n_patches}), skipping")
                    continue

                # === KEY OPTIMIZATION: Only compute loss on masked patches ===
                # Flatten and get masked indices
                mask_flat = mask_global.reshape(-1)  # [B*2 * num_patches]
                masked_indices = mask_flat.bool()
                
                # Count masked patches
                n_masked = masked_indices.sum().item()
                
                if n_masked == 0:
                    continue
                
                # Flatten patches
                t_flat = t_patches.reshape(-1, out_dim)  # [B*2 * num_patches, out_dim]
                s_flat = s_patches_global.reshape(-1, out_dim)  # [B*2 * num_patches, out_dim]
                
                # Verify dimensions match before indexing
                if t_flat.shape[0] != mask_flat.shape[0]:
                    print(f"Warning: t_flat shape {t_flat.shape[0]} != mask_flat shape {mask_flat.shape[0]}, skipping")
                    continue
                
                # Select only masked patches (memory efficient!)
                t_masked = t_flat[masked_indices]  # [n_masked, out_dim]
                s_masked = s_flat[masked_indices]  # [n_masked, out_dim]
                
                # Compute softmax for teacher (with centering)
                with torch.no_grad():
                    t_soft = F.softmax((t_masked - self.center_patches.detach()) / temp, dim=-1)
                
                # Compute log_softmax for student
                s_log = F.log_softmax(s_masked / self.student_temp, dim=-1)
                
                # Compute cross-entropy loss
                patch_loss = torch.sum(-t_soft * s_log, dim=-1)  # [n_masked]
                
                total_patch_loss += patch_loss.sum()
                total_masked_count += n_masked

            # Average over all masked patches
            if total_masked_count > 0:
                mim_loss = total_patch_loss / total_masked_count

        self.update_center(teacher_output, teacher_patch_out)

        # Weighted combination of CLS and MIM losses
        total_loss = cls_loss + self.mim_loss_weight * mim_loss

        return total_loss, cls_loss, mim_loss

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


class MaskGenerator:
    """
    Generate random block masks for MIM (Masked Image Modeling).
    
    This implements block-wise masking similar to BEiT/iBOT, which masks
    contiguous blocks of patches rather than random individual patches.
    
    Args:
        input_size (int): Size of input image (assumes square)
        patch_size (int): Size of each patch
        mask_ratio (float): Ratio of patches to mask (0.0 to 1.0)
        min_num_patches (int): Minimum number of patches in a mask block
        max_num_patches (int): Maximum number of patches in a mask block
        min_aspect (float): Minimum aspect ratio of mask blocks
        max_aspect (float): Maximum aspect ratio of mask blocks
    """
    def __init__(
        self,
        input_size=224,
        patch_size=16,
        mask_ratio=0.4,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches_per_side = input_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.mask_ratio = mask_ratio
        
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches if max_num_patches else int(self.num_patches * 0.5)
        
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect if max_aspect else 1 / min_aspect
        
        self.log_aspect_ratio = (math.log(self.min_aspect), math.log(self.max_aspect))
    
    def __call__(self):
        """Generate a mask. Returns tensor of shape [num_patches] with 1s for masked positions."""
        mask = torch.zeros(self.num_patches_per_side, self.num_patches_per_side, dtype=torch.float32)
        mask_count = 0
        target_mask_count = int(self.num_patches * self.mask_ratio)
        
        attempts = 0
        max_attempts = 100
        
        while mask_count < target_mask_count and attempts < max_attempts:
            attempts += 1
            
            # Random block size
            target_area = torch.empty(1).uniform_(self.min_num_patches, self.max_num_patches).item()
            aspect_ratio = math.exp(torch.empty(1).uniform_(*self.log_aspect_ratio).item())
            
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if h < self.num_patches_per_side and w < self.num_patches_per_side:
                top = torch.randint(0, self.num_patches_per_side - h + 1, (1,)).item()
                left = torch.randint(0, self.num_patches_per_side - w + 1, (1,)).item()
                
                # Count new masked patches (avoid double counting)
                new_mask = torch.zeros_like(mask)
                new_mask[top:top+h, left:left+w] = 1.0
                newly_masked = ((new_mask == 1) & (mask == 0)).sum().item()
                
                if mask_count + newly_masked <= target_mask_count * 1.1:  # Allow slight overshoot
                    mask[top:top+h, left:left+w] = 1.0
                    mask_count = mask.sum().item()
        
        # If we couldn't reach target with blocks, randomly fill remaining
        if mask_count < target_mask_count:
            remaining = target_mask_count - int(mask_count)
            flat_mask = mask.reshape(-1)
            unmasked_indices = (flat_mask == 0).nonzero(as_tuple=True)[0]
            if len(unmasked_indices) > 0:
                perm = torch.randperm(len(unmasked_indices))[:remaining]
                flat_mask[unmasked_indices[perm]] = 1.0
                mask = flat_mask.reshape(self.num_patches_per_side, self.num_patches_per_side)
        
        return mask.reshape(-1)  # [num_patches]
    
    def get_num_patches(self):
        return self.num_patches


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    
    Now with MIM support: generates masks for global crops.
    
    Args:
        backbone: Vision transformer backbone
        head: DINO projection head
        mask_ratio (float): Ratio of patches to mask for MIM (default: 0.4)
        mask_global_crops_only (bool): Only mask global crops (default: True)
    """
    def __init__(self, backbone, head, mask_ratio=0.4, mask_global_crops_only=True):
        super(MultiCropWrapper, self).__init__()
        # Disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self.mask_ratio = mask_ratio
        self.mask_global_crops_only = mask_global_crops_only
        
        # Cache mask generators for different resolutions
        self._mask_generators = {}
    
    def _get_mask_generator(self, input_size, patch_size=16):
        """Get or create mask generator for given input size."""
        key = (input_size, patch_size)
        if key not in self._mask_generators:
            self._mask_generators[key] = MaskGenerator(
                input_size=input_size,
                patch_size=patch_size,
                mask_ratio=self.mask_ratio,
            )
        return self._mask_generators[key]

    def forward(self, x, marker_ids=None, is_student=True):
        """
        Forward pass with optional masking for MIM.
        
        Args:
            x: List of input tensors at different resolutions
            marker_ids: Optional marker IDs for each crop
            is_student: If True and training, generate masks for MIM
            
        Returns:
            Dict containing:
                - x_norm_clstoken: CLS token outputs through head
                - x_norm_patchtokens: Dict of patch tokens grouped by num_patches
                - masks: Dict of masks grouped by num_patches (only if is_student=True)
        """
        # Convert to list if not already
        if not isinstance(x, list):
            x = [x]
        
        # Get unique image sizes and their counts
        sizes = torch.tensor([inp.shape[-1] for inp in x])
        unique_sizes, counts = torch.unique_consecutive(sizes, return_counts=True)
        
        start_idx = 0
        cls_outputs = []
        backbone_cls_global = []  # backbone CLS tokens for global crops (for KoLeo)
        patch_outputs_by_size = {}  # Group by number of patches
        masks_by_size = {}

        # Track which resolution index corresponds to global crops
        # Assuming global crops come first and have the largest size
        global_crop_size = sizes[0].item()  # First crop is global
        
        for size_idx, (size, count) in enumerate(zip(unique_sizes, counts)):
            end_idx = start_idx + count
            size_val = size.item()
            
            # Concatenate crops of the same resolution
            _out = torch.cat(x[start_idx:end_idx])
            batch_size_total = _out.shape[0]
            
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

            # Collect backbone CLS tokens for global crops (used for KoLeo in student)
            if is_student and size_val == global_crop_size:
                backbone_cls_global.append(backbone_out["x_norm_clstoken"])

            # Collect patch tokens grouped by number of patches
            if "x_norm_patchtokens" in backbone_out and backbone_out["x_norm_patchtokens"] is not None:
                patches = backbone_out["x_norm_patchtokens"]
                num_patches = patches.shape[1]
                
                if num_patches not in patch_outputs_by_size:
                    patch_outputs_by_size[num_patches] = []
                patch_outputs_by_size[num_patches].append(patches)
                
                # Generate masks for MIM (only for student, and only for global crops if configured)
                if is_student and self.mask_ratio > 0:
                    # Determine if we should mask this resolution
                    is_global_crop = (size_val == global_crop_size)
                    should_mask = is_global_crop or (not self.mask_global_crops_only)
                    
                    if should_mask:
                        # Get patch size from backbone (default 16)
                        patch_size = getattr(self.backbone, 'patch_size', 16)
                        if hasattr(patch_size, '__iter__'):
                            patch_size = patch_size[0]
                        
                        mask_gen = self._get_mask_generator(size_val, patch_size)
                        
                        # Generate mask for each image in batch
                        masks_list = []
                        for _ in range(batch_size_total):
                            mask = mask_gen()  # [num_patches_from_generator]
                            masks_list.append(mask)
                        
                        generated_masks = torch.stack(masks_list).to(_out.device)  # [B, num_patches_gen]
                        
                        # Verify mask num_patches matches actual num_patches from backbone
                        if generated_masks.shape[1] != num_patches:
                            # Resize mask if needed (can happen with different patch sizes)
                            # Use simple interpolation: repeat or truncate
                            if generated_masks.shape[1] < num_patches:
                                # Repeat mask to match
                                repeat_factor = (num_patches + generated_masks.shape[1] - 1) // generated_masks.shape[1]
                                generated_masks = generated_masks.repeat(1, repeat_factor)[:, :num_patches]
                            else:
                                # Truncate
                                generated_masks = generated_masks[:, :num_patches]
                        
                        if num_patches not in masks_by_size:
                            masks_by_size[num_patches] = []
                        masks_by_size[num_patches].append(generated_masks)
            
            start_idx = end_idx
        
        # Concatenate all CLS token outputs and pass through head
        cls_concat = torch.cat(cls_outputs)
        head_output = self.head(cls_concat)
        
        # Prepare return dictionary
        ret = {
            "x_norm_clstoken": head_output,
        }

        # Return backbone CLS tokens for global crops (for KoLeo loss in student)
        if backbone_cls_global:
            ret["backbone_cls_tokens"] = torch.cat(backbone_cls_global, dim=0)
        
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
        
        # Process masks grouped by size (concatenate masks for same num_patches)
        if masks_by_size:
            mask_dict = {}
            for num_patches, mask_list in masks_by_size.items():
                mask_dict[num_patches] = torch.cat(mask_list, dim=0)
            ret["masks"] = mask_dict
        else:
            ret["masks"] = None
        
        return ret