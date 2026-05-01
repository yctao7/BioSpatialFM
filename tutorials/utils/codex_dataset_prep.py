import numpy as np
import tifffile
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import re
import os
import math
import pandas as pd
import h5py
import logging

logging.getLogger('tifffile').setLevel(logging.ERROR)

def compute_stats(input_dir, output_dir, level=3):
    if os.path.exists(os.path.join(output_dir, 'marker_info.csv')):
        df_stats = pd.read_csv(os.path.join(output_dir, 'marker_info.csv'))
        return df_stats
    os.makedirs(output_dir, exist_ok=True)
    stats = defaultdict(lambda: {'n': 0, 'mean': 0, 'm2': 0})
    file_path_list = [file_path for file_path in Path(input_dir).rglob("*") if file_path.is_file()]
    for file_path in tqdm(file_path_list):
        try:
            tif = tifffile.TiffFile(file_path)
        except:
            print(f'BROKEN FILE: {file_path}')
            continue
        series = tif.series[0]
        dtype = series.dtype
        C, H, W = series.shape
        actual_level = min(level, len(series.levels) - 1)

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

        for i, marker in enumerate(tqdm(markers, leave=False)):
            try:
                chan_data = series.levels[actual_level].asarray(key=i).astype(np.float64)
            except Exception as e:
                print(f"\n[解压失败] 文件损坏: {file_path.name} | 通道: {marker} | 错误: {e}")
                continue
            # Always normalize to [0,1] so uint8 / uint16 donors contribute on the same scale.
            # Matches the dtype-aware normalization done at training time in data_augmentation.Normalization.
            chan_data /= np.iinfo(dtype).max
            stats[marker] = update_welford(stats[marker], chan_data)
    
    df_stats = get_final_stats(stats)
    df_stats.to_csv(os.path.join(output_dir, 'marker_info.csv'), index=False)
    return df_stats


def update_welford(existing_stats, new_data):
    """
    使用并行/批量 Welford 算法更新统计量
    new_data: 一个 numpy array (单通道像素值)
    """
    na = existing_stats['n']
    mu_a = existing_stats['mean']
    m2_a = existing_stats['m2']

    new_data = new_data[new_data != 0]
    nb = new_data.size
    if nb == 0:
            return existing_stats

    mu_b = np.mean(new_data)
    m2_b = np.sum((new_data - mu_b) ** 2)

    if na == 0:
        return {'n': nb, 'mean': mu_b, 'm2': m2_b}

    n_total = na + nb
    delta = mu_b - mu_a
    
    new_mean = mu_a + delta * (nb / n_total)
    new_m2 = m2_a + m2_b + (delta ** 2) * (na * nb / n_total)
    
    return {'n': n_total, 'mean': new_mean, 'm2': new_m2}


def get_final_stats(stats):
    # 1. 获取所有 Marker 名字并按字母顺序排序
    sorted_marker_names = sorted(stats.keys())
    
    final_data = []
    
    # 2. 遍历排序后的名字，分配新的 channel_id (0, 1, 2...)
    for idx, name in enumerate(sorted_marker_names):
        s = stats[name]
        n = s['n']
        
        # 计算均值和标准差
        mean = s['mean']
        # 标准差公式: sqrt(m2 / n)
        std = math.sqrt(s['m2'] / n) if n > 0 else 0
        
        final_data.append({
            'channel_id': idx,
            'marker_name': name.upper(),
            'marker_mean': mean,
            'marker_std': std
        })
    
    # 3. 转换为 DataFrame
    df = pd.DataFrame(final_data)
    return df