"""
Dataset classes for loading multiplex images for KRONOS fine-tuning.
"""

import os
import torch
import numpy as np
import tifffile
from torch.utils.data import Dataset
from typing import Optional, List, Tuple, Union
import pandas as pd
import h5py


class MultiplexImageDataset(Dataset):
    """
    Dataset for loading multiplex images from TIFF files.
    
    Args:
        image_paths: List of paths to multiplex image TIFF files
        marker_ids: List of marker IDs for each sample (optional)
        transform: Data augmentation transform (optional)
        mean_values: Mean values for normalization (optional)
        std_values: Std values for normalization (optional)
    """
    def __init__(
        self,
        image_paths: List[str],
        marker_ids: Optional[List[List[int]]] = None,
        transform=None,
        mean_values: Optional[np.ndarray] = None,
        std_values: Optional[np.ndarray] = None,
    ):
        self.image_paths = image_paths
        self.marker_ids = marker_ids
        self.transform = transform
        self.mean_values = mean_values
        self.std_values = std_values
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load multiplex image
        img_path = self.image_paths[idx]
        image = tifffile.imread(img_path)  # Shape: [C, H, W] or [H, W, C]
        
        # Ensure channel-first format
        if image.ndim == 3 and image.shape[-1] < image.shape[0]:
            # Likely [H, W, C], transpose to [C, H, W]
            image = np.transpose(image, (2, 0, 1))
        
        # Convert to torch tensor
        image = torch.from_numpy(image).float()
        
        # Normalize if mean and std are provided
        if self.mean_values is not None and self.std_values is not None:
            mean = torch.tensor(self.mean_values).view(-1, 1, 1)
            std = torch.tensor(self.std_values).view(-1, 1, 1)
            image = (image - mean) / (std + 1e-8)
        
        # Get marker IDs
        if self.marker_ids is not None:
            marker_id = self.marker_ids[idx]
        else:
            # Default marker IDs starting from 4
            marker_id = [i + 4 for i in range(image.shape[0])]
        
        # Apply augmentation
        if self.transform is not None:
            crops = self.transform(image)
            return crops, marker_id
        
        return image, marker_id


class MultiplexImageFolderDataset(Dataset):
    """
    Dataset for loading multiplex images from a folder structure.
    Each subfolder contains marker-specific TIFF files.
    
    Expected folder structure:
        data_root/
            sample_1/
                marker_1.tiff
                marker_2.tiff
                ...
            sample_2/
                marker_1.tiff
                marker_2.tiff
                ...
    
    Args:
        data_root: Root directory containing sample folders
        marker_names: List of marker names to load (optional, loads all if None)
        transform: Data augmentation transform (optional)
        mean_values: Mean values for normalization (optional)
        std_values: Std values for normalization (optional)
    """
    def __init__(
        self,
        data_root: str,
        marker_names: Optional[List[str]] = None,
        transform=None,
        mean_values: Optional[np.ndarray] = None,
        std_values: Optional[np.ndarray] = None,
    ):
        self.data_root = data_root
        self.marker_names = marker_names
        self.transform = transform
        self.mean_values = mean_values
        self.std_values = std_values
        
        # Find all sample folders
        self.sample_folders = [
            os.path.join(data_root, d) 
            for d in os.listdir(data_root) 
            if os.path.isdir(os.path.join(data_root, d))
        ]
        self.sample_folders.sort()
        
    def __len__(self):
        return len(self.sample_folders)
    
    def __getitem__(self, idx):
        sample_folder = self.sample_folders[idx]
        
        # Get marker files
        if self.marker_names is not None:
            marker_files = [
                os.path.join(sample_folder, f"{name}.tiff")
                for name in self.marker_names
            ]
        else:
            # Load all TIFF files in the folder
            marker_files = [
                os.path.join(sample_folder, f)
                for f in sorted(os.listdir(sample_folder))
                if f.endswith(('.tiff', '.tif'))
            ]
        
        # Load markers
        markers = []
        for marker_file in marker_files:
            if os.path.exists(marker_file):
                marker_img = tifffile.imread(marker_file)
                markers.append(marker_img)
            else:
                print(f"Warning: Marker file not found: {marker_file}")
        
        # Stack markers into multi-channel image
        if len(markers) > 0:
            image = np.stack(markers, axis=0)  # Shape: [C, H, W]
        else:
            raise ValueError(f"No markers found in {sample_folder}")
        
        # Convert to torch tensor
        image = torch.from_numpy(image).float()
        
        # Normalize if mean and std are provided
        if self.mean_values is not None and self.std_values is not None:
            mean = torch.tensor(self.mean_values).view(-1, 1, 1)
            std = torch.tensor(self.std_values).view(-1, 1, 1)
            image = (image - mean) / (std + 1e-8)
        
        # Default marker IDs
        marker_ids = [i + 4 for i in range(image.shape[0])]
        
        # Apply augmentation
        if self.transform is not None:
            crops = self.transform(image)
            return crops, marker_ids
        
        return image, marker_ids


class MultiplexPatchDataset(Dataset):
    """
    Dataset for loading pre-extracted multiplex image patches.
    
    Args:
        patch_dir: Directory containing patch numpy files
        patch_list: List of patch filenames (optional)
        transform: Data augmentation transform (optional)
        mean_values: Mean values for normalization (optional)
        std_values: Std values for normalization (optional)
    """
    def __init__(
        self,
        patch_dir: str,
        patch_list: Optional[List[str]] = None,
        transform=None,
    ):
        self.patch_dir = patch_dir
        self.transform = transform
        
        if patch_list is not None:
            self.patch_files = patch_list
        else:
            self.patch_files = [
                f for f in os.listdir(patch_dir)
                if f.endswith('.h5')
            ]
            self.patch_files.sort()
        
    def __len__(self):
        return len(self.patch_files)
    
    def __getitem__(self, idx):
        patch_file = self.patch_files[idx]
        patch_path = os.path.join(self.patch_dir, patch_file)
        
        with h5py.File(patch_path, 'r') as f:
            patch = {m: f[m][()] for m in f}
        
        # Apply augmentation
        if self.transform is not None:
            crops, marker_ids = self.transform(patch)
            return crops, marker_ids
        else:
            raise ValueError("Transform must be provided for patch dataset.")


def load_marker_metadata(metadata_path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Load marker metadata including names, mean, and std values.
    
    Args:
        metadata_path: Path to marker metadata CSV file
        
    Returns:
        Tuple of (marker_names, mean_values, std_values)
    """
    df = pd.read_csv(metadata_path)
    marker_metadata = df.set_index('marker_name').to_dict(orient='index')
    return marker_metadata


def collate_fn_multicrop(batch):
    """
    Custom collate function for multi-crop batches.
    
    Args:
        batch: List of (crops, marker_ids) tuples
        
    Returns:
        Tuple of (crops_list, marker_ids_list)
    """
    # Separate crops and marker_ids
    all_crops = [item[0] for item in batch]
    all_marker_ids = [item[1] for item in batch]
    
    # Check if crops is a list (multi-crop) or single tensor
    if isinstance(all_crops[0], list):
        # Multi-crop case
        num_crops = len(all_crops[0])
        crops_list = []
        
        for crop_idx in range(num_crops):
            # Stack all samples for this crop
            crop_batch = torch.stack([crops[crop_idx] for crops in all_crops])
            crops_list.append(crop_batch)
        
        # Replicate marker_ids for each crop
        marker_ids_list = []
        for marker_ids in all_marker_ids:
            marker_ids_tensor = torch.tensor(marker_ids)
            # Repeat for each crop
            for _ in range(num_crops):
                marker_ids_list.append(marker_ids_tensor)
        
        return crops_list, marker_ids_list
    else:
        # Single image case
        crops_batch = torch.stack(all_crops)
        marker_ids_list = [torch.tensor(ids) for ids in all_marker_ids]
        return crops_batch, marker_ids_list
