"""
Utility for extracting patches from large multiplex TIFF images.
"""

import os
import numpy as np
import tifffile
from pathlib import Path
from typing import Optional, Tuple, List
from tqdm import tqdm
import xml.etree.ElementTree as ET
import re
import h5py
import argparse
import logging

logging.getLogger('tifffile').setLevel(logging.ERROR)


class PatchExtractor:
    """
    Extract fixed-size patches from large multiplex TIFF images.
    
    Args:
        patch_size: Size of patches to extract (default: 256)
        stride: Stride for patch extraction (default: None, uses patch_size for non-overlapping)
        min_valid_pixels: Minimum number of valid (non-zero) pixels required (default: None)
    """
    def __init__(
        self,
        patch_size: int = 256,
        stride: Optional[int] = None,
        min_valid_pixels: Optional[int] = None,
    ):
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.min_valid_pixels = min_valid_pixels
        
    def extract_patches_from_image(
        self,
        image: np.ndarray,
    ) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Extract patches from a single image with smart edge handling.
        
        For edge patches that would extend beyond image boundaries, they are
        "bounced back" from the edge to ensure full coverage without padding.
        
        Args:
            image: Multiplex image array with shape [C, H, W] or [H, W, C]
            
        Returns:
            List of (patch, (row, col)) tuples
        """
        # Ensure channel-first format [C, H, W]
        if image.ndim == 2:
            # Single channel image, add channel dimension
            image = image[np.newaxis, ...]
        elif image.ndim == 3:
            if image.shape[-1] < image.shape[0] and image.shape[-1] < image.shape[1]:
                # Likely [H, W, C], transpose to [C, H, W]
                image = np.transpose(image, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        num_channels, height, width = image.shape
        patches = []
        
        # Calculate row positions with smart edge handling
        row_positions = []
        row_start = 0
        while row_start + self.patch_size <= height:
            row_positions.append(row_start)
            row_start += self.stride
        
        # If there's remaining space, add a patch bounced back from the edge
        if row_positions[-1] + self.patch_size < height:
            row_positions.append(height - self.patch_size)
        
        # Calculate column positions with smart edge handling
        col_positions = []
        col_start = 0
        while col_start + self.patch_size <= width:
            col_positions.append(col_start)
            col_start += self.stride
        
        # If there's remaining space, add a patch bounced back from the edge
        if col_positions[-1] + self.patch_size < width:
            col_positions.append(width - self.patch_size)
        
        # Extract patches at calculated positions
        for i, row_start in enumerate(row_positions):
            for j, col_start in enumerate(col_positions):
                row_end = row_start + self.patch_size
                col_end = col_start + self.patch_size
                
                # Extract patch
                patch = image[:, row_start:row_end, col_start:col_end]
                
                # Check if patch has enough valid pixels
                if self.min_valid_pixels is not None:
                    valid_pixels = np.sum(patch > 0)
                    if valid_pixels < self.min_valid_pixels:
                        continue
                
                patches.append((patch, (i, j)))
        
        return patches
    
    def extract_patches_from_tiff(
        self,
        tiff_path: str,
        output_dir: str,
        prefix: Optional[str] = None,
        save_format: str = 'npy',
    ) -> int:
        """
        Extract patches from a TIFF file and save them.
        
        Args:
            tiff_path: Path to input TIFF file
            output_dir: Directory to save patches
            prefix: Prefix for patch filenames (default: uses input filename)
            save_format: Format to save patches ('npy' or 'tiff')
            
        Returns:
            Number of patches extracted
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load TIFF image
        print(f"Loading image from {tiff_path}...")
        image = tifffile.imread(tiff_path)
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        
        # Extract patches
        print(f"Extracting patches (size={self.patch_size}, stride={self.stride})...")
        patches = self.extract_patches_from_image(image)
        
        # Generate prefix if not provided
        if prefix is None:
            prefix = Path(tiff_path).stem
        
        # Save patches
        print(f"Saving {len(patches)} patches to {output_dir}...")
        for idx, (patch, (row, col)) in enumerate(tqdm(patches)):
            if save_format == 'npy':
                filename = f"{prefix}_patch_{idx:04d}_r{row}_c{col}.npy"
                filepath = os.path.join(output_dir, filename)
                np.save(filepath, patch.astype(np.float32))
            elif save_format == 'tiff':
                filename = f"{prefix}_patch_{idx:04d}_r{row}_c{col}.tiff"
                filepath = os.path.join(output_dir, filename)
                tifffile.imwrite(filepath, patch)
            else:
                raise ValueError(f"Unsupported save format: {save_format}")
        
        print(f"✓ Extracted and saved {len(patches)} patches")
        return len(patches)
    
    def extract_patches_from_folder(
        self,
        input_folder: str,
        output_dir: str,
        pattern: str = "*.tif*",
        save_format: str = 'npy',
    ) -> int:
        """
        Extract patches from all TIFF files in a folder.
        
        Args:
            input_folder: Folder containing TIFF files
            output_dir: Directory to save patches
            pattern: File pattern to match (default: "*.tif*")
            save_format: Format to save patches ('npy' or 'tiff')
            
        Returns:
            Total number of patches extracted
        """
        # Find all TIFF files
        tiff_files = list(Path(input_folder).glob(pattern))
        print(f"Found {len(tiff_files)} TIFF files")
        
        total_patches = 0
        for tiff_file in tiff_files:
            print(f"\n{'='*60}")
            print(f"Processing: {tiff_file.name}")
            print('='*60)
            
            num_patches = self.extract_patches_from_tiff(
                str(tiff_file),
                output_dir,
                prefix=tiff_file.stem,
                save_format=save_format,
            )
            total_patches += num_patches
        
        print(f"\n{'='*60}")
        print(f"✓ Total patches extracted: {total_patches}")
        print('='*60)
        
        return total_patches
    
    def extract_patches_with_overlap_handling(
        self,
        tiff_path: str,
        output_dir: str,
        prefix: Optional[str] = None,
        save_format: str = 'npy',
        padding_mode: str = 'reflect',
    ) -> int:
        """
        Extract patches with padding to ensure full image coverage.
        
        Args:
            tiff_path: Path to input TIFF file
            output_dir: Directory to save patches
            prefix: Prefix for patch filenames
            save_format: Format to save patches ('npy' or 'tiff')
            padding_mode: Padding mode for edge patches ('reflect', 'constant', 'edge')
            
        Returns:
            Number of patches extracted
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load TIFF image
        print(f"Loading image from {tiff_path}...")
        image = tifffile.imread(tiff_path)
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        
        # Ensure channel-first format [C, H, W]
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        elif image.ndim == 3:
            if image.shape[-1] < image.shape[0] and image.shape[-1] < image.shape[1]:
                image = np.transpose(image, (2, 0, 1))
        
        num_channels, height, width = image.shape
        
        # Calculate padding needed
        pad_height = (self.patch_size - (height % self.patch_size)) % self.patch_size
        pad_width = (self.patch_size - (width % self.patch_size)) % self.patch_size
        
        # Pad image if necessary
        if pad_height > 0 or pad_width > 0:
            print(f"Padding image: height +{pad_height}, width +{pad_width}")
            if padding_mode == 'constant':
                image = np.pad(
                    image,
                    ((0, 0), (0, pad_height), (0, pad_width)),
                    mode='constant',
                    constant_values=0
                )
            else:
                image = np.pad(
                    image,
                    ((0, 0), (0, pad_height), (0, pad_width)),
                    mode=padding_mode
                )
        
        # Extract patches
        print(f"Extracting patches (size={self.patch_size}, stride={self.stride})...")
        patches = self.extract_patches_from_image(image)
        
        # Generate prefix if not provided
        if prefix is None:
            prefix = Path(tiff_path).stem
        
        # Save patches
        print(f"Saving {len(patches)} patches to {output_dir}...")
        for idx, (patch, (row, col)) in enumerate(tqdm(patches)):
            if save_format == 'npy':
                filename = f"{prefix}_patch_{idx:04d}_r{row}_c{col}.npy"
                filepath = os.path.join(output_dir, filename)
                np.save(filepath, patch.astype(np.float32))
            elif save_format == 'tiff':
                filename = f"{prefix}_patch_{idx:04d}_r{row}_c{col}.tiff"
                filepath = os.path.join(output_dir, filename)
                tifffile.imwrite(filepath, patch)
        
        print(f"✓ Extracted and saved {len(patches)} patches")
        return len(patches)


def extract_patches(input_dir, output_dir):
    patch_extractor = PatchExtractor(patch_size=256, stride=256)

    file_path_list = [file_path for file_path in Path(input_dir).rglob("*") if file_path.is_file()]
    pbar = tqdm(file_path_list)
    for file_path in pbar:
        pbar.set_description(f"Processing {file_path}")
        try:
            tif = tifffile.TiffFile(file_path)
        except:
            print(f'BROKEN FILE: {file_path}')
            continue
        series = tif.series[0]
        dtype = series.dtype
        C, H, W = series.shape

        if tif.is_ome:
            xml_str = tif.ome_metadata
            root = ET.fromstring(xml_str)
            # OME 标准的命名空间
            ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
            # 提取所有通道的名字
            markers = [c.get('Name') for c in root.findall('.//ome:Channel', ns)]
        else:
            markers = []
            for page in tif.series[0].pages:
                match = re.search(r'<Biomarker>(.*?)</Biomarker>', page.description)
                markers.append(match.group(1))
        assert C == len(markers)

        marker_list, chan_data_list = [], []
        for i, marker in enumerate(markers):
            try:
                chan_data = series.levels[0].asarray(key=i)
                marker_list.append(marker)
                chan_data_list.append(chan_data)
            except Exception as e:
                print(f"\n[解压失败] 文件损坏: {file_path.name} | 通道: {marker} | 错误: {e}")
                continue

        patches = patch_extractor.extract_patches_from_image(np.stack(chan_data_list, axis=0))
        os.makedirs(output_dir, exist_ok=True)
        for idx, (patch, (row, col)) in enumerate(patches):
            if patch.sum() == 0:
                continue
            with h5py.File(os.path.join(output_dir, f"{file_path.stem}_{row:03d}_{col:03d}.h5"), 'w') as f:
                for i, marker in enumerate(marker_list):
                    f.create_dataset(marker.upper(), data=patch[i, :, :])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="KRONOS 数据预处理：从多通道 TIFF 提取 Patch 并保存为 H5")
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='包含原始 TIFF 文件的输入目录路径')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='保存提取出的 .h5 patch 的输出目录路径')
    args = parser.parse_args()
    extract_patches(args.input_dir, args.output_dir)