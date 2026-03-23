from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

import numpy as np
from PIL import Image


SUPPORTED_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
}


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model_id: int
    model_name: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class ImageEntry:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def _read_exact(fid, num_bytes: int) -> bytes:
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise ValueError(f"Unexpected EOF while reading {num_bytes} bytes")
    return data


def _unpack(fid, fmt: str):
    return struct.unpack("<" + fmt, _read_exact(fid, struct.calcsize("<" + fmt)))


def _read_c_string(fid) -> str:
    chunks = bytearray()
    while True:
        ch = fid.read(1)
        if ch == b"":
            raise ValueError("Unexpected EOF while reading image name")
        if ch == b"\x00":
            return chunks.decode("utf-8")
        chunks.extend(ch)


def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    norm = np.linalg.norm(qvec)
    if norm == 0.0:
        raise ValueError("Encountered zero-norm quaternion in images.bin")
    qw, qx, qy, qz = qvec / norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _read_cameras_bin(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with path.open("rb") as fid:
        num_cameras = _unpack(fid, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _unpack(fid, "iiQQ")
            if model_id not in SUPPORTED_CAMERA_MODELS:
                raise ValueError(
                    f"Unsupported camera model id {model_id} in {path}. "
                    f"Supported models: SIMPLE_PINHOLE, PINHOLE"
                )
            model_name, num_params = SUPPORTED_CAMERA_MODELS[model_id]
            params = np.array(_unpack(fid, "d" * num_params), dtype=np.float64)
            cameras[camera_id] = Camera(
                camera_id=camera_id,
                model_id=model_id,
                model_name=model_name,
                width=width,
                height=height,
                params=params,
            )
    return cameras


def _read_images_bin(path: Path) -> list[ImageEntry]:
    images: list[ImageEntry] = []
    with path.open("rb") as fid:
        num_images = _unpack(fid, "Q")[0]
        for _ in range(num_images):
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _unpack(fid, "idddddddi")
            name = _read_c_string(fid)
            num_points2d = _unpack(fid, "Q")[0]
            fid.seek(24 * num_points2d, 1)
            images.append(
                ImageEntry(
                    image_id=image_id,
                    qvec=np.array([qw, qx, qy, qz], dtype=np.float64),
                    tvec=np.array([tx, ty, tz], dtype=np.float64),
                    camera_id=camera_id,
                    name=name,
                )
            )
    return images


def _read_points3d_bin(path: Path) -> np.ndarray:
    points = []
    with path.open("rb") as fid:
        num_points = _unpack(fid, "Q")[0]
        for _ in range(num_points):
            _, x, y, z, _, _, _, _ = _unpack(fid, "QdddBBBd")
            track_length = _unpack(fid, "Q")[0]
            fid.seek(8 * track_length, 1)
            points.append((x, y, z))
    return np.asarray(points, dtype=np.float64)


def _camera_intrinsics(camera: Camera) -> tuple[float, float, float, float]:
    if camera.model_name == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params.tolist()
        return f, f, cx, cy
    if camera.model_name == "PINHOLE":
        fx, fy, cx, cy = camera.params.tolist()
        return fx, fy, cx, cy
    raise ValueError(
        f"Unsupported camera model {camera.model_name}. Supported models: SIMPLE_PINHOLE, PINHOLE"
    )


def _depth_output_path(output_dirpath: Path, image_name: str) -> Path:
    image_relpath = Path(image_name)
    if image_relpath.suffix:
        return output_dirpath / image_relpath.with_suffix(".tiff")
    return output_dirpath / Path(str(image_relpath) + ".tiff")


def _project_depth_map(points_world: np.ndarray, image: ImageEntry, camera: Camera) -> np.ndarray:
    rotation = _qvec_to_rotmat(image.qvec)
    points_cam = points_world @ rotation.T + image.tvec

    z = points_cam[:, 2]
    valid = z > 0.0
    if not np.any(valid):
        return np.zeros((camera.height, camera.width), dtype=np.float32)

    points_cam = points_cam[valid]
    z = z[valid]

    fx, fy, cx, cy = _camera_intrinsics(camera)
    x = points_cam[:, 0] / z
    y = points_cam[:, 1] / z
    u = fx * x + cx
    v = fy * y + cy

    px = np.floor(u + 0.5).astype(np.int64)
    py = np.floor(v + 0.5).astype(np.int64)

    in_bounds = (px >= 0) & (px < camera.width) & (py >= 0) & (py < camera.height)
    if not np.any(in_bounds):
        return np.zeros((camera.height, camera.width), dtype=np.float32)

    px = px[in_bounds]
    py = py[in_bounds]
    z = z[in_bounds]

    depth = np.full((camera.height, camera.width), np.inf, dtype=np.float64)
    flat_indices = py * camera.width + px
    np.minimum.at(depth.ravel(), flat_indices, z)
    depth[~np.isfinite(depth)] = 0.0
    return depth.astype(np.float32)


def generate_depth_maps(scene_root: str, output_dirpath: str | None = None) -> None:
    scene_root_path = Path(scene_root)
    sparse_dir = scene_root_path / "sparse" / "0"
    cameras_bin = sparse_dir / "cameras.bin"
    images_bin = sparse_dir / "images.bin"
    points3d_bin = sparse_dir / "points3D.bin"

    required_files = [cameras_bin, images_bin, points3d_bin]
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required COLMAP sparse model files under "
            f"{sparse_dir}: {', '.join(missing_files)}"
        )

    cameras = _read_cameras_bin(cameras_bin)
    images = _read_images_bin(images_bin)
    points_world = _read_points3d_bin(points3d_bin)

    output_root = Path(output_dirpath) if output_dirpath is not None else scene_root_path / "depths"
    output_root.mkdir(parents=True, exist_ok=True)

    for image in tqdm(images, desc="Projecting depth maps"):
        if image.camera_id not in cameras:
            raise ValueError(
                f"Image {image.name} (id={image.image_id}) references missing camera_id {image.camera_id}"
            )
        camera = cameras[image.camera_id]
        depth = _project_depth_map(points_world, image, camera)
        output_path = _depth_output_path(output_root, image.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(depth, mode="F").save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project COLMAP sparse points to image planes and save float32 depth TIFF files."
    )
    parser.add_argument("--scene_root", required=True, help="Scene root containing sparse/0")
    parser.add_argument(
        "--output_dirpath",
        default=None,
        help="Directory to write depth TIFF files. Defaults to <scene_root>/depths",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_depth_maps(scene_root=args.scene_root, output_dirpath=args.output_dirpath)
