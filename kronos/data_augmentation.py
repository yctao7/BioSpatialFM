# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import List, Optional
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random


class GaussianNoise:
    """Add Gaussian noise to the image."""
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class RandomRotation90:
    """Randomly rotate the image by 0, 90, 180, or 270 degrees."""
    def __call__(self, img):
        k = random.choice([0, 1, 2, 3])
        return torch.rot90(img, k, [1, 2])


class MarkerSelection:
    """
    Per-sample channel selection for the DINO branch.

    `fixed_markers=['DAPI', 'DNA']` lets each panel find its own nuclear/structural
    anchor: CODEX panels contain DAPI but no DNA, IMC panels contain DNA but no
    DAPI. The matching loop picks the FIRST entry that's in the panel, so each
    panel contributes one anchor (not two), and the missing-fixed compensation
    keeps the total channel count consistent with the legacy `fixed=['DAPI'],
    n_random=2` behavior.

    For CODEX: 1 fixed (DAPI) + 1 missing-comp + 1 random  = 3 channels (with anchor)
    For IMC:   1 fixed (DNA)  + 1 missing-comp + 1 random  = 3 channels (with anchor)
    Both panels get an anchor. Pre-canonical-name fix (when IMC h5 keys still had
    isotope prefixes) IMC was getting 0 fixed + 3 random, no structural anchor.
    """
    def __init__(self, fixed_markers=None, n_random_markers=1):
        if fixed_markers is None:
            fixed_markers = ['DAPI', 'DNA']
        self.fixed_markers = list(fixed_markers)
        self.n_random_markers = n_random_markers

    def __call__(self, image):
        image_new = {}
        # Include fixed markers that exist; count how many are missing.
        n_missing_fixed = 0
        for marker in self.fixed_markers:
            if marker in image:
                image_new[marker] = image[marker]
            else:
                n_missing_fixed += 1
        # Total = len(fixed_markers) + n_random_markers; missing fixed entries
        # are compensated with extra random picks, so a 2-entry fixed list still
        # yields 3 channels even when only 1 entry actually matches.
        n_random = min(self.n_random_markers + n_missing_fixed,
                       len(image) - len(image_new))
        available_markers = [m for m in image.keys() if m not in image_new]
        selected_markers = random.sample(available_markers, n_random)
        for marker in selected_markers:
            image_new[marker] = image[marker]
        return image_new


class Normalization:
    def __init__(self, marker_metadata, add_noise=True, noise_std=0.1):
        self.marker_metadata = marker_metadata
        self.add_noise = add_noise
        self.noise_std = noise_std

    def __call__(self, image):
        """
        Normalize multiplex image.

        Args:
            image: {marker_name: tensor} where each tensor is of shape [H, W]
        """
        image_new = np.stack([image[m] for m in image], axis=0)

        dtype = image_new.dtype
        max_val = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
        image_new = torch.from_numpy(image_new.astype(np.float32) / max_val)

        means, stds, marker_ids = [], [], []
        for m in image:
            meta = self.marker_metadata.get(m, {})
            means.append(meta.get('marker_mean', 0.0))
            stds.append(meta.get('marker_std', 1.0))
            marker_ids.append(meta.get('marker_id', 0))
        mean = torch.tensor(means).view(-1, 1, 1)
        std = torch.tensor(stds).view(-1, 1, 1)
        image_new = (image_new - mean) / (std + 1e-8)

        if self.add_noise:
            image_new = image_new + torch.randn_like(image_new) * self.noise_std

        return image_new, marker_ids


class ApplyToImageOnly:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, input_data):
        img, extra = input_data
        img_augmented = self.transform(img)
        return img_augmented, extra


class DataAugmentationDINO:
    """
    Data augmentation for DINO self-supervised learning on multiplex images.
    Creates multiple crops with different augmentations for each image.
    
    Args:
        global_crops_scale (tuple): Scale range for global crops
        local_crops_scale (tuple): Scale range for local crops
        local_crops_number (int): Number of local crops to generate
        global_crops_size (int): Size of global crops
        local_crops_size (int): Size of local crops
    """
    def __init__(
        self,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=8,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        
        # Geometric transformations for multiplex images
        # Note: We avoid color jittering as markers have specific meanings
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crops_size, 
                scale=global_crops_scale, 
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRotation90(),
        ])
        
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crops_size, 
                scale=global_crops_scale, 
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRotation90(),
        ])
        
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crops_size, 
                scale=local_crops_scale, 
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRotation90(),
        ])

    def __call__(self, image):
        """
        Apply augmentation to multiplex image.
        
        Args:
            image: Input tensor of shape [C, H, W] where C is number of markers
            
        Returns:
            List of augmented crops [global_crop1, global_crop2, local_crop1, ..., local_cropN]
        """
        crops = []
        
        # Generate 2 global crops
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        
        # Generate multiple local crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
            
        return crops


class MultiViewDataAugmentation:
    """
    Simplified multi-view augmentation for multiplex images.
    Creates two augmented views of the same image.
    """
    def __init__(self, crop_size=224, scale=(0.4, 1.0)):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                crop_size, 
                scale=scale, 
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRotation90(),
        ])
        
    def __call__(self, image):
        """Generate two augmented views."""
        view1 = self.transform(image)
        view2 = self.transform(image)
        return [view1, view2]


class MaskConsistencyAugmentation:
    """
    Per-sample full-panel view for the channel-masking consistency loss (Lmask).

    Returns ONE normalized tensor with all markers (channels in canonical sorted
    order so samples sharing a marker panel land in the same panel group at
    collation time). The DinoMaskCollator does the per-group channel masking at
    batch level -- matching kronos's forward-time convention that all samples in
    a (sub-)batch share the same marker_ids tensor.

    A single bilinear resize to `target_size` (default 256) is applied so that
    samples sharing a marker panel can be stacked at collation time. CODEX / IMC
    patches are already 256x256, but the islet source (peterszj/islet_patches_h5)
    has variable per-islet bounding-box sizes (~80 to ~5000), and a same-panel
    CODEX+islet mix would otherwise fail torch.stack in DinoMaskCollator. Bilinear
    is mean-preserving and produces no negative values for the non-negative
    multiplex inputs. No additive Gaussian noise either (would otherwise inject a
    separate invariance into Lmask).
    """
    def __init__(self, marker_metadata, target_size: int = 256):
        self.marker_metadata = marker_metadata
        self.target_size = int(target_size)
        # No additive noise: channel set is the only systematic teacher/student difference.
        self.normalize = Normalization(marker_metadata, add_noise=False)

    def __call__(self, image):
        markers = sorted(image.keys())
        if len(markers) < 2:
            raise ValueError(
                f"Sample has only {len(markers)} marker(s); mask consistency requires >= 2."
            )
        ordered = {m: image[m] for m in markers}
        tensor, marker_ids = self.normalize(ordered)  # tensor: [C, H, W], marker_ids: List[int]
        if tensor.shape[-2] != self.target_size or tensor.shape[-1] != self.target_size:
            tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False,
                antialias=True,
            ).squeeze(0)
        return tensor, marker_ids


class CombinedDinoMaskTransform:
    """
    Returns BOTH the standard DINO multicrop output AND a single full-panel mask view
    from one h5 read. Output structure:
        ((dino_crops, dino_marker_ids), (mask_tensor, mask_marker_ids))
    The dataset returns this tuple as-is; DinoMaskCollator groups by panel and
    performs the per-group channel masking that produces teacher/student tensors.

    The mask view is just the raw normalized patch (no spatial augmentation) -- see
    MaskConsistencyAugmentation for rationale.
    """
    def __init__(
        self,
        marker_metadata,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=8,
        global_crops_size=224,
        local_crops_size=96,
    ):
        # DINO branch: identical to the existing 'dino' pipeline behavior.
        self.dino_pipeline = transforms.Compose([
            MarkerSelection(),
            Normalization(marker_metadata),
            ApplyToImageOnly(DataAugmentationDINO(
                global_crops_scale=global_crops_scale,
                local_crops_scale=local_crops_scale,
                local_crops_number=local_crops_number,
                global_crops_size=global_crops_size,
                local_crops_size=local_crops_size,
            )),
        ])
        self.mask_pipeline = MaskConsistencyAugmentation(marker_metadata)

    def __call__(self, image):
        # MarkerSelection / Normalization in the DINO branch build new dicts/tensors and do
        # not mutate `image`, so the mask branch can safely consume the same source dict.
        dino_out = self.dino_pipeline(image)
        mask_out = self.mask_pipeline(image)
        return dino_out, mask_out


def get_augmentation_pipeline(
    marker_metadata,
    augmentation_type='dino',
    global_crops_size=224,
    local_crops_size=96,
    local_crops_number=8,
    global_crops_scale=(0.4, 1.0),
    local_crops_scale=(0.05, 0.4),
):
    """
    Get data augmentation pipeline.
    
    Args:
        augmentation_type: Type of augmentation ('dino' or 'simple')
        global_crops_size: Size of global crops
        local_crops_size: Size of local crops
        local_crops_number: Number of local crops
        global_crops_scale: Scale range for global crops
        local_crops_scale: Scale range for local crops
        
    Returns:
        Augmentation transform
    """
    if augmentation_type == 'dino':
        return transforms.Compose([
            MarkerSelection(),
            Normalization(marker_metadata),
            ApplyToImageOnly(DataAugmentationDINO(
                global_crops_scale=global_crops_scale,
                local_crops_scale=local_crops_scale,
                local_crops_number=local_crops_number,
                global_crops_size=global_crops_size,
                local_crops_size=local_crops_size,
            ))
        ])
    elif augmentation_type == 'simple':
        return MultiViewDataAugmentation(
            crop_size=global_crops_size,
            scale=global_crops_scale,
        )
    elif augmentation_type == 'dino+mask':
        return CombinedDinoMaskTransform(
            marker_metadata,
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            local_crops_number=local_crops_number,
            global_crops_size=global_crops_size,
            local_crops_size=local_crops_size,
        )
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
