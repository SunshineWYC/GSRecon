import os
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from gaussian_splatting.utils.graphics_utils import BasicPointCloud
from datasets.colmap_reader import read_points3D_binary, read_points3D_text


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
