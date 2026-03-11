import os
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
from datasets.colmap_reader import read_points3D_binary, read_points3D_text
import torch


def fetch_ply(path, scene_scale=1.0):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T * scene_scale
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def load_pcdfile(pcd_filepath, scene_scale=1.0):
    if pcd_filepath.endswith(".bin"):
        xyzs, rgbs, _ = read_points3D_binary(pcd_filepath)
        pcd = BasicPointCloud(points=xyzs*scene_scale, colors=rgbs/255.0, normals=None)
    elif pcd_filepath.endswith(".txt"):
        xyzs, rgbs, _ = read_points3D_text(pcd_filepath)
        pcd = BasicPointCloud(points=xyzs*scene_scale, colors=rgbs/255.0, normals=None)
    elif pcd_filepath.endswith(".ply"):
        plydata = PlyData.read(pcd_filepath)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T * scene_scale
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        pcd = BasicPointCloud(points=positions, colors=colors, normals=None)
    else:
        raise "pcd_filepath should be .bin or .txt generated from COLMAP-SFM, or .ply format"
    return pcd


def infinite_dataloader(loader):
    while True:
        for batch in loader:
            yield batch


def collate_single_view(batch):
    """
    Collate function for batch_size=1 that avoids torch.stack copies.
    Expects items like: (view_id, view_data_dict).
    """
    if len(batch) != 1:
        raise ValueError(f"collate_single_view expects batch_size=1, got batch size {len(batch)}.")
    view_id, view_data = batch[0]

    collated = {}
    for key, value in view_data.items():
        if isinstance(value, torch.Tensor):
            collated[key] = value.unsqueeze(0)
        else:
            collated[key] = value

    return view_id, collated


def create_dataloader(dataset, batch_size=1, shuffle=False, num_workers=4, preload=False, preload_device="cpu"):
    """
    Encapsulate DataLoader creation logic and optimize parameters based on preloading settings.
    
    Args:
        dataset: Dataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        num_workers: Number of data loading workers.
        preload: Whether to enable preloading.
        preload_device: Preloading device (str or torch.device).
    """
    preload = bool(preload)
    if isinstance(preload_device, torch.device):
        is_cuda_preload = preload_device.type == "cuda"
    else:
        is_cuda_preload = str(preload_device).lower().startswith("cuda")

    gpu_preload = preload and is_cuda_preload

    # Optimized configuration for GPU preloading
    if gpu_preload:
        nw, pm, pw = 0, False, False
    # Optimized configuration for CPU preloading
    elif preload:
        nw, pm, pw = 0, True, False
    # Regular loading configuration
    else:
        nw, pm, pw = num_workers, True, (num_workers > 0)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=pw,
        drop_last=False,
        collate_fn=collate_single_view,
    )
