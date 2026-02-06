import torch
import gsplat
from torch import Tensor
import torch.nn.functional as F


class GSplatRenderer:
    def __init__(self, render_mode="RGB+ED", camera_model="pinhole", absgrad=True, **kwargs):
        self.render_mode = render_mode
        self.camera_model = camera_model
        self.absgrad = absgrad

        for key, value in kwargs.items():
            setattr(self, key, value)

    def render(self, gaussians, intrinsic, extrinsic, sh_degree, image_height, image_width, device="cuda:0"):
        """
        Docstring for gsplatRenderer
        
        :param gaussians: GaussianModel
        :param intrinsic: camera intrinsic tensor on GPU, shape [1, 3, 3]
        :param extrinsic: camera extrinsic tensor on GPU, shape [1, 4, 4]
        :param sh_degree: sh_degree in rendering
        :param image_height: image height
        :param image_width: image width
        :param device: rendering device
        :param camera_model: camera model type, default is "pinhole"

        :return: 
        - image_rendered: rendered image tensor on GPU, shape [3, H, W], RGB values in [0, 1]
        - depth_rendered: rendered depth tensor on GPU, shape [H, W], depth values
        - info: additional rendering info dict
        """

        render_colors, render_alphas, info = gsplat.rendering.rasterization(
            means=gaussians.get_xyz,
            quats=gaussians.get_rotation,
            scales=gaussians.get_scaling,
            opacities=gaussians.get_opacity.squeeze(),
            colors=gaussians.get_features,
            viewmats=extrinsic.to(device),
            Ks=intrinsic.to(device),
            sh_degree=sh_degree,
            width=image_width,
            height=image_height,
            camera_model=self.camera_model,
            render_mode=self.render_mode,
            absgrad=self.absgrad,   # densification with absolute gradient, values in info
        )

        image_rendered = render_colors[0, :, :, :3].permute(2, 0, 1)
        depth_rendered = render_colors[0, :, :, 3]

        return image_rendered, depth_rendered, info


class GSplatCameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, view_ids):
        super().__init__()
        self.view_id_to_embed = {int(view_id): idx for idx, view_id in enumerate(view_ids)}
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(len(view_ids), 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, view_ids) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            view_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        view_ids_list = [int(v) for v in view_ids]
        embed_list = []
        for vid in view_ids_list:
            if vid not in self.view_id_to_embed:
                raise KeyError(f"view_id {vid} is not in CameraOptModule mapping.")
            embed_list.append(self.view_id_to_embed[vid])

        embed_indices = torch.tensor(embed_list, device=camtoworlds.device, dtype=torch.long)
        assert camtoworlds.shape[:-2] == embed_indices.shape
        batch_dims = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_indices)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_dims, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_dims, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class GSplatPoseRefiner:
    def __init__(self, view_ids, device, total_iters, pose_updater_params, pose_optimizer_params):
        self.device = device
        self.pose_updater = GSplatCameraOptModule(view_ids).to(device)

        pose_updater_params = pose_updater_params or {}
        init_type = pose_updater_params.get("init", "zero")
        if init_type == "random":
            std = float(pose_updater_params.get("std", 0.01))
            self.pose_updater.random_init(std)
        else:
            self.pose_updater.zero_init()

        pose_optimizer_params = pose_optimizer_params or {}
        self.start_iter = max(1, int(pose_optimizer_params.get("start_iter", 5000)))
        self.end_iter = int(pose_optimizer_params.get("end_iter", 25000))
        lr = float(pose_optimizer_params.get("lr", 1e-5))
        reg = float(pose_optimizer_params.get("reg", 1e-6))
        self.gamma_final_ratio = float(pose_optimizer_params.get("gamma_final_ratio", 0.01))

        self.pose_optimizer = torch.optim.Adam(self.pose_updater.parameters(), lr=lr, weight_decay=reg)

        effective_end = min(self.end_iter, int(total_iters))
        self.update_steps = max(0, effective_end - self.start_iter + 1)
        if self.update_steps > 0:
            gamma = self.gamma_final_ratio ** (1.0 / self.update_steps)
            self.pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.pose_optimizer, gamma=gamma)
        else:
            self.pose_scheduler = None

    def refine_extrinsic_w2c(self, extrinsic_w2c: Tensor, view_ids, iteration: int) -> Tensor:
        if iteration < self.start_iter:
            return extrinsic_w2c

        camtoworld = torch.linalg.inv(extrinsic_w2c)
        if iteration <= self.end_iter:
            camtoworld = self.pose_updater(camtoworld, view_ids)
        else:
            with torch.no_grad():
                camtoworld = self.pose_updater(camtoworld, view_ids)
        return torch.linalg.inv(camtoworld)

    def step(self, iteration: int):
        if self.start_iter <= iteration <= self.end_iter and self.update_steps > 0:
            self.pose_optimizer.step()
            if self.pose_scheduler is not None:
                self.pose_scheduler.step()
        self.pose_optimizer.zero_grad(set_to_none=True)

    def export_refined_train_w2c(self, train_dataset, device, scene_scale):
        refined_train_w2c = {}
        with torch.no_grad():
            for view_id, view_info in zip(train_dataset.view_ids, train_dataset.views_info_list):
                w2c = torch.tensor(view_info.extrinsic, dtype=torch.float32, device=device).unsqueeze(0)
                w2c[..., :3, 3] *= scene_scale
                w2c_ref = self.refine_extrinsic_w2c(w2c, [int(view_id)], self.end_iter + 1)
                w2c_ref = w2c_ref.squeeze(0)
                w2c_ref[:3, 3] /= scene_scale
                refined_train_w2c[int(view_id)] = w2c_ref.detach().cpu().numpy()
        return refined_train_w2c
