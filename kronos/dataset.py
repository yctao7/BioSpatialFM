"""
Dataset classes for loading multiplex images for KRONOS fine-tuning.
"""

import math
import os
import random
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
        recursive: bool = False,
    ):
        self.patch_dir = patch_dir
        self.transform = transform

        if patch_list is not None:
            # patch_list contains absolute paths
            self.patch_files = patch_list
        elif recursive:
            # Walk all subdirectories
            self.patch_files = []
            for root, _, files in os.walk(patch_dir):
                for f in files:
                    if f.endswith('.h5'):
                        self.patch_files.append(os.path.join(root, f))
            self.patch_files.sort()
        else:
            self.patch_files = [
                os.path.join(patch_dir, f)
                for f in os.listdir(patch_dir)
                if f.endswith('.h5')
            ]
            self.patch_files.sort()

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        patch_path = self.patch_files[idx]

        with h5py.File(patch_path, 'r') as f:
            patch = {m: f[m][()] for m in f}

        if self.transform is None:
            raise ValueError("Transform must be provided for patch dataset.")
        # Pass through whatever the transform produces; the matching collate_fn handles it.
        # For the standard 'dino' pipeline this is the (crops, marker_ids) tuple expected by
        # collate_fn_multicrop. For 'dino+mask' it is (dino_out, mask_out) consumed by
        # DinoMaskCollator.
        return self.transform(patch)


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

        # marker_ids_list shape: [num_crops][B], crop-major. Each inner list has one
        # 1D marker_ids tensor per sample so MultiCropWrapper.forward can hand the
        # *per-sample* tensors straight to apply_masks instead of broadcasting one
        # sample's tensor across the whole sub-batch.
        sample_marker_tensors = [torch.tensor(ids) for ids in all_marker_ids]
        marker_ids_list = [list(sample_marker_tensors) for _ in range(num_crops)]

        return crops_list, marker_ids_list
    else:
        # Single image case
        crops_batch = torch.stack(all_crops)
        marker_ids_list = [torch.tensor(ids) for ids in all_marker_ids]
        return crops_batch, marker_ids_list


class DinoMaskCollator:
    """
    Collate function for the combined DINO + channel-mask consistency pipeline.

    Each batch element is `(dino_out, mask_out)` where:
      - dino_out  = (crops_list, marker_ids) -- standard DINO multicrop output
      - mask_out  = (full_tensor[C, H, W], marker_ids[C])  -- canonical (sorted) order

    Variable-C handling:
      - Samples are GROUPED by their marker panel (keyed on the canonical marker_ids
        tuple). All samples in a group have identical marker order, so we can stack
        them into [Bg, C, H, W] and use a single shared marker_ids tensor for the
        whole group -- matching MultiCropWrapper's per-(sub-)batch broadcast convention.

    K student views per teacher view (K = `n_student_views`):
      - For each panel group, K independent student views are produced by sampling
        K independent keep-fractions s_k ~ U[keep_min, keep_max] and K independent
        random channel subsets of size floor(s_k * C). All K views match the same
        single teacher full-panel view (K-to-1 pairing per sample).
      - This gives the mask-CE loss K times more gradient signal per sample with
        no extra teacher cost (teacher forward is single, no_grad).
      - Naming mirrors DINO's `local_crops_scale=(0.05, 0.4)` which expresses a
        fraction-to-keep range. So `keep_min/max` are the channel analog: at
        keep=0.05 only 5% of channels survive, at 0.4 forty percent do.

    Forced-keep markers + minimum kept channels:
      - `always_keep_marker_ids` (e.g. DAPI for CODEX, DNA for IMC) lists markers
        that MUST appear in every student view. The teacher view is always the
        full panel (unaffected). Names absent from a panel are silently ignored.
      - `min_kept_channels` is a floor on the student view size (default 3 to
        match the DINO branch's 3-channel input). It prevents degenerate K_s=1
        or K_s=2 when keep_min is small (e.g. 0.05) and panel size is moderate.
    """
    def __init__(self, keep_min: float = 0.05, keep_max: float = 0.4,
                 always_keep_marker_ids=None, n_student_views: int = 2,
                 min_kept_channels: int = 3):
        assert 0.0 < keep_min <= keep_max < 1.0
        assert n_student_views >= 1
        assert min_kept_channels >= 1
        self.keep_min = keep_min
        self.keep_max = keep_max
        self.always_keep_marker_ids = set(always_keep_marker_ids) if always_keep_marker_ids else set()
        self.n_student_views = int(n_student_views)
        self.min_kept_channels = int(min_kept_channels)

    def _build_student_view(self, teacher_imgs, full_ids, key, forced_keep_idx, free_idx, C):
        """One sampling + slicing of a student view from the teacher tensor."""
        keep = random.uniform(self.keep_min, self.keep_max)
        K_s_target = int(math.floor(keep * C))
        # Floors: at least min_kept_channels, at least len(forced_keep_idx);
        # cap: strictly less than C so the student is never identical to teacher.
        K_s = max(self.min_kept_channels, len(forced_keep_idx),
                  min(K_s_target, C - 1))
        n_free = K_s - len(forced_keep_idx)
        free_pick = random.sample(free_idx, n_free) if n_free > 0 else []
        student_idx = sorted(set(forced_keep_idx) | set(free_pick))
        idx_t = torch.tensor(student_idx, dtype=torch.long)
        student_imgs = teacher_imgs.index_select(1, idx_t)            # [Bg, K_s, H, W]
        student_ids = full_ids.index_select(0, idx_t)                 # [K_s]
        return student_imgs, student_ids, K_s, keep

    def __call__(self, batch):
        dino_part = [s[0] for s in batch]
        mask_part = [s[1] for s in batch]

        dino_crops, dino_marker_ids = collate_fn_multicrop(dino_part)

        by_panel = {}
        for tensor, ids in mask_part:
            key = tuple(ids)
            by_panel.setdefault(key, []).append(tensor)

        mask_groups = []
        for key in sorted(by_panel.keys()):
            tensors = by_panel[key]
            teacher_imgs = torch.stack(tensors)                       # [Bg, C, H, W]
            C = teacher_imgs.shape[1]
            full_ids = torch.tensor(list(key), dtype=torch.long)      # [C]

            forced_keep_idx = sorted(i for i, mid in enumerate(key)
                                     if mid in self.always_keep_marker_ids)
            free_idx = [i for i in range(C) if i not in forced_keep_idx]

            student_views = []  # list of K dicts: {'imgs', 'ids', 'K_s', 'keep'}
            for _ in range(self.n_student_views):
                s_imgs, s_ids, K_s, keep_k = self._build_student_view(
                    teacher_imgs, full_ids, key, forced_keep_idx, free_idx, C
                )
                student_views.append({'imgs': s_imgs, 'ids': s_ids, 'K_s': K_s, 'keep': keep_k})

            mask_groups.append({
                'teacher_imgs': teacher_imgs,
                'teacher_ids': full_ids,
                'student_views': student_views,
                'C': C,
            })

        return {
            'dino_crops': dino_crops,
            'dino_marker_ids': dino_marker_ids,
            'mask_groups': mask_groups,
        }
