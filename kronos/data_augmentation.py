# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import List, Optional
import torch
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
    def __init__(self, fixed_markers=['DAPI'], n_random_markers=2):
        self.fixed_markers = fixed_markers
        self.n_random_markers = n_random_markers

    def __call__(self, image):
        image_new = {}
        for marker in self.fixed_markers:
            image_new[marker] = image[marker]
        available_markers = [m for m in image.keys() if m not in self.fixed_markers]
        selected_markers = random.sample(available_markers, self.n_random_markers)
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

        image_new = torch.from_numpy(image_new.astype(np.float32) / np.iinfo(image_new.dtype).max)

        mean = torch.tensor([self.marker_metadata[m]['marker_mean'] for m in image]).view(-1, 1, 1)
        std = torch.tensor([self.marker_metadata[m]['marker_std'] for m in image]).view(-1, 1, 1)
        image_new = (image_new - mean) / (std + 1e-8)
        
        if self.add_noise:
            image_new = image_new + torch.randn_like(image_new) * self.noise_std
        
        marker_ids = [self.marker_metadata[m]['marker_id'] for m in image]
            
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
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
