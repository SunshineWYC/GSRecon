import os
import cv2
import torch
import numpy as np
from datasets.sensors import PerspectiveViewInfo
from datasets.colmap_reader import (
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat
)
from concurrent.futures import ThreadPoolExecutor, as_completed


class COLMAPSceneInfo:
    """
    Information about a COLMAP scene, initial data read from colmap format output without loading images.
    """
    def __init__(self, scene_dirpath):
        self.scene_dirpath = scene_dirpath
        assert os.path.exists(os.path.join(scene_dirpath, "sparse")), "sparse fold is not exist while the scene_type is colmap."

        try:
            cameras_extrinsic_file = os.path.join(scene_dirpath, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(scene_dirpath, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(scene_dirpath, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(scene_dirpath, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        self.image_dirpath = os.path.join(scene_dirpath, "images")
        self.views_info = self.load_views_info(cam_extrinsics, cam_intrinsics)
        self.view_ids = list(self.views_info.keys())

        if os.path.exists(os.path.join(scene_dirpath, "sparse/0", "points3D.bin")):
            self.pcd_filepath = os.path.join(scene_dirpath, "sparse/0", "points3D.bin")
        elif os.path.exists(os.path.join(scene_dirpath, "sparse/0", "points3D.txt")):
            self.pcd_filepath = os.path.join(scene_dirpath, "sparse/0", "points3D.txt")
        elif os.path.exists(os.path.join(scene_dirpath, "sparse/0", "points3D.ply")):
            self.pcd_filepath = os.path.join(scene_dirpath, "sparse/0", "points3D.ply")
        else:
            self.pcd_filepath = None

    def load_views_info(self, cam_extrinsics, cam_intrinsics):
        views_info = {}
    
        for index, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            view_id = extr.id
            intr = cam_intrinsics[extr.camera_id]
            image_height = intr.height
            image_width = intr.width

            R = qvec2rotmat(extr.qvec)  # W2C
            T = np.array(extr.tvec)   # W2C

            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = focal_length_x
                cx = intr.params[1]
                cy = intr.params[2]
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                cx = intr.params[2]
                cy = intr.params[3]
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            extrinsic = np.zeros((4, 4), dtype=np.float32)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = T
            extrinsic[3, 3] = 1.0

            intrinsic = np.array([[focal_length_x, 0, cx],
                                [0, focal_length_y, cy],
                                [0, 0, 1]], dtype=np.float32)

            views_info[view_id] = PerspectiveViewInfo(
                view_id=view_id,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                image_height=image_height,
                image_width=image_width,
                image_filepath = os.path.join(self.image_dirpath, extr.name)
            )
        return views_info


class COLMAPDataset(torch.utils.data.Dataset):
    def __init__(self, views_info, image_scale=1.0, scene_scale=1.0, preload=False, split=None, eval_split_interval=8):
        super(COLMAPDataset, self).__init__()
        self.views_info = views_info
        self.image_scale = image_scale
        self.scene_scale = scene_scale

        self.views_info_list, self.view_ids, self.num_views = self._split_views(split, split_interval=eval_split_interval)

        self.preload = preload
        if preload:
            self.views_data = self._preload_data()

    def __len__(self):
        return self.num_views

    def _split_views(self, split, split_interval=8):
        if split == "train":
            view_ids = [view_id for idx, view_id in enumerate(self.views_info.keys()) if idx % split_interval != 0]
            views_info_list = [view_info for idx, view_info in enumerate(self.views_info.values()) if idx % split_interval != 0]
        elif split == "val":
            view_ids = [view_id for idx, view_id in enumerate(self.views_info.keys()) if idx % split_interval == 0]
            views_info_list = [view_info for idx, view_info in enumerate(self.views_info.values()) if idx % split_interval == 0]
        else:
            view_ids = list(self.views_info.keys())
            views_info_list = list(self.views_info.values())
        
        return views_info_list, view_ids, len(views_info_list)

    def _read_image(self, image_filepath):
        image = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_filepath}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        new_width = int(width * self.image_scale)
        new_height = int(height * self.image_scale)
        if new_width <= 0 or new_height <= 0:
            raise ValueError(f"Invalid scaled image size ({new_width}, {new_height}) for scale {self.image_scale}")
        if new_width != width or new_height != height:
            interp = cv2.INTER_AREA if self.image_scale < 1.0 else cv2.INTER_LINEAR
            image = cv2.resize(image, (new_width, new_height), interpolation=interp)
        np_image = image.astype(np.float32) / 255.0
        return np_image[:, :, :3], new_height, new_width

    def _read_mask(self, mask_filepath):
        mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_filepath}")
        height, width = mask.shape[:2]
        new_width = int(width * self.image_scale)
        new_height = int(height * self.image_scale)
        if new_width <= 0 or new_height <= 0:
            raise ValueError(f"Invalid scaled mask size ({new_width}, {new_height}) for scale {self.image_scale}")
        if new_width != width or new_height != height:
            mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        np_mask = (mask > 128).astype(np.float32)  # binary mask, 1.0 valid, 0.0 masked
        return np_mask

    def _load_single_view_data(self, view_info):
        image, new_height, new_width = self._read_image(view_info.image_filepath)

        height_scale = new_height / view_info.image_height
        width_scale = new_width / view_info.image_width
        intrinsic = view_info.intrinsic.copy()
        intrinsic[:1, :] *= width_scale
        intrinsic[1:2, :] *= height_scale

        extrinsic = view_info.extrinsic.copy()  # [4, 4]
        extrinsic[:3, 3] *= self.scene_scale

        intrinsic = torch.tensor(intrinsic, dtype=torch.float32)
        extrinsic = torch.tensor(extrinsic, dtype=torch.float32)
        image = torch.tensor(image).permute(2, 0, 1) # [C, H, W]

        # Handle case when mask file doesn't exist
        if view_info.mask_filepath is not None and os.path.exists(view_info.mask_filepath):
            mask = self._read_mask(view_info.mask_filepath)
            mask = torch.tensor(mask, dtype=torch.float32)  # [H, W]
        else:
            # Create a default mask (all valid regions)
            mask = torch.ones(new_height, new_width, dtype=torch.float32)
        
        view_data = {
            "image": image,
            "mask": mask,
            "height": new_height,
            "width": new_width,
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "image_filepath": view_info.image_filepath,
            "camera_model": "pinhole",
        }

        return view_data

    def _preload_data(self, num_workers=8):
        """
        Preload all snapshot data into memory with multi-threading.
        """
        views_data = [None] * self.num_views
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._load_single_view_data, view_info): idx for idx, view_info in enumerate(self.views_info_list)}
            for future in as_completed(futures):
                idx = futures[future]
                views_data[idx] = future.result()
        return views_data

    def __getitem__(self, idx):
        idx = idx % self.num_views

        if not self.preload:
            return self.view_ids[idx], self._load_single_view_data(self.views_info_list[idx])
        else:
            return self.view_ids[idx], self.views_data[idx]



if __name__ == "__main__":
    # Example usage
    colmap_scene = COLMAPSceneInfo("./data/dorm_1007_002")
    colmap_dataset = COLMAPDataset(
        colmap_scene.views_info, 
        image_scale=1.0, 
        scene_scale=1.0, 
        preload=True,
        split="train"
    )

    dataloader = torch.utils.data.DataLoader(
        colmap_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        drop_last=False, 
        pin_memory=True
    )
    
    dataloader_iter = iter(dataloader)
    view_id, view_data = next(dataloader_iter)

    print("Hello world!")
