import os
import numpy as np
from PIL import Image


class PerspectiveViewInfo:
    def __init__(self, view_id, intrinsic, extrinsic, image_height, image_width, cam_id=0, **kwargs):
        self.view_id = view_id
        self.intrinsic = intrinsic  # 3x3 numpy array
        self.extrinsic = extrinsic  # 4x4 numpy array, W2C
        self.image_height = image_height
        self.image_width = image_width
        self.cam_id = cam_id

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        return None
    
    def load_image(self):
        assert self.image_filepath is not None, "Image filepath is None!"
        if not os.path.isfile(self.image_filepath):
            raise RuntimeError('%s does not exist!' % self.image_filepath)
        image = Image.open(self.image_filepath)
        return np.array(image)

    def project_points(self, points):
        """
        Project 3D points(in world coordinate) to 2D image plane using the camera intrinsic and extrinsic parameters.
        """
        # points: Nx3 numpy array
        assert points.shape[1] == 3, "Input points should be Nx3 numpy array."
        num_points = points.shape[0]

        # Convert points to homogeneous coordinates
        points_homogeneous = np.hstack((points, np.ones((num_points, 1))))  # Nx4

        # Transform points from world coordinate to camera coordinate
        points_camera = np.dot(self.extrinsic, points_homogeneous.T).T  # Nx4

        # Project points to image plane
        points_image_homogeneous = np.dot(self.intrinsic, points_camera[:, :3].T).T  # Nx3
        depth = points_camera[:, 2]  # N
        # Normalize to get pixel coordinates
        points_image = points_image_homogeneous[:, :2] / points_image_homogeneous[:, 2:3]  # Nx2
        u, v = np.round(points_image[:, 0]).astype(int), np.round(points_image[:, 1]).astype(int)

        return u, v, depth

    def get_sparse_depth(self, points):
        """
        Get sparse depth map from 3D points(in world coordinate).
        """
        u, v, depth = self.project_points(points)
        sparse_depth = np.zeros((self.image_height, self.image_width), dtype=np.float32)

        for i in range(len(u)):
            if 0 <= u[i] < self.image_width and 0 <= v[i] < self.image_height and depth[i] > 0:
                if sparse_depth[v[i], u[i]] == 0 or sparse_depth[v[i], u[i]] > depth[i]:
                    sparse_depth[v[i], u[i]] = depth[i]

        return sparse_depth
