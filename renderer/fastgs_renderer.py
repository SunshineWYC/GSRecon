import math
import torch
from torch import Tensor

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except Exception as e:
    GaussianRasterizationSettings = None
    GaussianRasterizer = None
    _DGR_IMPORT_ERROR = e


def _get_projection_matrix(znear, zfar, fovx, fovy, device, dtype):
    tan_half_fov_y = math.tan(fovy / 2.0)
    tan_half_fov_x = math.tan(fovx / 2.0)

    top = tan_half_fov_y * znear
    bottom = -top
    right = tan_half_fov_x * znear
    left = -right

    P = torch.zeros(4, 4, device=device, dtype=dtype)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def _skew_sym_mat(x: Tensor) -> Tensor:
    ssm = torch.zeros(3, 3, device=x.device, dtype=x.dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def _so3_exp(theta: Tensor) -> Tensor:
    W = _skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=theta.device, dtype=theta.dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    return I + (torch.sin(angle) / angle) * W + ((1 - torch.cos(angle)) / (angle ** 2)) * W2


def _V(theta: Tensor) -> Tensor:
    I = torch.eye(3, device=theta.device, dtype=theta.dtype)
    W = _skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        return I + 0.5 * W + (1.0 / 6.0) * W2
    return I + W * ((1.0 - torch.cos(angle)) / (angle ** 2)) + W2 * ((angle - torch.sin(angle)) / (angle ** 3))


def _se3_exp(tau: Tensor) -> Tensor:
    rho = tau[:3]
    theta = tau[3:]
    R = _so3_exp(theta)
    t = _V(theta) @ rho
    T = torch.eye(4, device=tau.device, dtype=tau.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class FastGSRenderer:
    def __init__(
        self,
        render_mode="RGB+ED",
        camera_model="pinhole",
        absgrad=True,
        znear=0.01,
        zfar=100.0,
        antialiasing=False,
        scale_modifier=1.0,
        bg_color=None,
        **kwargs,
    ):
        if GaussianRasterizationSettings is None or GaussianRasterizer is None:
            raise ImportError(
                "diff_gaussian_rasterization is not available. "
                "Please install it in the current environment."
            ) from _DGR_IMPORT_ERROR

        self.znear = znear
        self.zfar = zfar
        self.antialiasing = antialiasing
        self.scale_modifier = scale_modifier
        self.bg_color = bg_color

        for key, value in kwargs.items():
            setattr(self, key, value)

    def render(
        self,
        gaussians,
        intrinsic,
        extrinsic,
        sh_degree,
        image_height,
        image_width,
        device="cuda:0",
        theta=None,
        rho=None,
    ):
        if GaussianRasterizationSettings is None or GaussianRasterizer is None:
            raise ImportError(
                "diff_gaussian_rasterization is not available. "
                "Please install it in the current environment."
            ) from _DGR_IMPORT_ERROR

        device = torch.device(device)
        extrinsic = extrinsic.to(device)
        intrinsic = intrinsic.to(device)

        fx = intrinsic[0, 0, 0].item()
        fy = intrinsic[0, 1, 1].item()
        tanfovx = image_width / (2.0 * fx)
        tanfovy = image_height / (2.0 * fy)
        fovx = 2.0 * math.atan(tanfovx)
        fovy = 2.0 * math.atan(tanfovy)

        viewmatrix = extrinsic[0].transpose(0, 1)
        projmatrix_raw = _get_projection_matrix(
            self.znear, self.zfar, fovx, fovy, device=viewmatrix.device, dtype=viewmatrix.dtype
        ).transpose(0, 1)
        projmatrix = viewmatrix @ projmatrix_raw
        campos = torch.inverse(viewmatrix)[3, :3]

        bg = self.bg_color
        if bg is None:
            bg = torch.zeros(3, device=device, dtype=viewmatrix.dtype)
        else:
            bg = torch.tensor(bg, device=device, dtype=viewmatrix.dtype)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(image_height),
            image_width=int(image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=self.scale_modifier,
            viewmatrix=viewmatrix,
            projmatrix=projmatrix,
            projmatrix_raw=projmatrix_raw,
            sh_degree=sh_degree,
            campos=campos,
            prefiltered=False,
            debug=False,
            antialiasing=self.antialiasing,
            get_flag=False,
            metric_map=torch.Tensor([]).to(torch.int32),
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gaussians.get_xyz
        means2D = torch.zeros(
            (means3D.shape[0], 6), dtype=means3D.dtype, device=means3D.device, requires_grad=True
        )
        try:
            means2D.retain_grad()
        except Exception:
            pass

        dc = gaussians.get_features_dc
        shs = gaussians.get_features_rest
        opacity = gaussians.get_opacity
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation

        render_colors, radii, invdepths, accum_metric_counts = rasterizer(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
            theta=theta,
            rho=rho,
        )

        image_rendered = render_colors.clamp(0, 1)  # [3, H, W]
        invdepths = invdepths.squeeze() # [H, W]
        info = {
            "radii": radii,
            "visibility_filter": (radii > 0).nonzero(),
            "means2d": means2D,
            "accum_metric_counts": accum_metric_counts,
            "height": int(image_height),
            "width": int(image_width),
            "n_cameras": int(extrinsic.shape[0]),
        }
        return image_rendered, invdepths, info


class FastGSCameraOptModule(torch.nn.Module):
    def __init__(self, view_ids):
        super().__init__()
        self.view_id_to_embed = {int(view_id): idx for idx, view_id in enumerate(view_ids)}
        self.rot_embed = torch.nn.Embedding(len(view_ids), 3)
        self.trans_embed = torch.nn.Embedding(len(view_ids), 3)

    def zero_init(self):
        torch.nn.init.zeros_(self.rot_embed.weight)
        torch.nn.init.zeros_(self.trans_embed.weight)

    def forward(self, view_ids):
        view_id = int(view_ids[0])
        if view_id not in self.view_id_to_embed:
            raise KeyError(f"view_id {view_id} is not in FastGSCameraOptModule mapping.")
        # Use shape [1] so embedding output is [1, 3], matching rasterizer backward grad shape.
        embed_idx = torch.tensor(
            [self.view_id_to_embed[view_id]], device=self.rot_embed.weight.device, dtype=torch.long
        )
        theta = self.rot_embed(embed_idx)
        rho = self.trans_embed(embed_idx)
        return theta, rho


class FastGSPoseRefiner:
    def __init__(self, view_ids, device, total_iters, pose_updater_params, pose_optimizer_params):
        self.device = device
        self.pose_updater = FastGSCameraOptModule(view_ids).to(device)

        pose_updater_params = pose_updater_params or {}
        init_type = pose_updater_params.get("init", "zero")
        if init_type == "zero":
            self.pose_updater.zero_init()
        else:
            self.pose_updater.zero_init()

        pose_optimizer_params = pose_optimizer_params or {}
        self.start_iter = max(1, int(pose_optimizer_params.get("start_iter", 2000)))
        self.end_iter = int(pose_optimizer_params.get("end_iter", 25000))
        rot_lr = float(pose_optimizer_params.get("rot_lr", 2e-5))
        trans_lr = float(pose_optimizer_params.get("trans_lr", 1e-5))
        reg = float(pose_optimizer_params.get("reg", 1e-6))
        gamma_final_ratio = float(pose_optimizer_params.get("gamma_final_ratio", 0.01))

        self.rot_optimizer = torch.optim.Adam(self.pose_updater.rot_embed.parameters(), lr=rot_lr, weight_decay=reg)
        self.trans_optimizer = torch.optim.Adam(self.pose_updater.trans_embed.parameters(), lr=trans_lr, weight_decay=reg)

        effective_end = min(self.end_iter, int(total_iters))
        self.update_steps = max(0, effective_end - self.start_iter + 1)
        if self.update_steps > 0:
            gamma = gamma_final_ratio ** (1.0 / self.update_steps)
            self.rot_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.rot_optimizer, gamma=gamma)
            self.trans_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.trans_optimizer, gamma=gamma)
        else:
            self.rot_scheduler = None
            self.trans_scheduler = None

    def _apply_delta_left_multiply(self, extrinsic_w2c: Tensor, theta: Tensor, rho: Tensor):
        with torch.no_grad():
            tau = torch.cat([rho.reshape(-1), theta.reshape(-1)], dim=0)
            T_delta = _se3_exp(tau)
            refined_w2c = extrinsic_w2c.clone()
            refined_w2c[0] = T_delta @ refined_w2c[0]
        return refined_w2c

    def _align_pose_delta(self, theta: Tensor, rho: Tensor, extrinsic_w2c: Tensor):
        if theta.numel() != 3 or rho.numel() != 3:
            raise ValueError(
                f"FastGSPoseRefiner expects theta/rho to contain 3 elements each (got {theta.shape=} {rho.shape=})."
            )
        theta = theta.reshape(1, 3).to(device=extrinsic_w2c.device, dtype=extrinsic_w2c.dtype)
        rho = rho.reshape(1, 3).to(device=extrinsic_w2c.device, dtype=extrinsic_w2c.dtype)
        return theta, rho

    def refine_pose(self, extrinsic_w2c: Tensor, view_ids, iteration: int):
        """
        Actually directly get the delta_rot(theta) and delta_trans(rho) here
        
        :param extrinsic_w2c: initial extrinsic_w2c
        :param view_ids: current view_id
        :param iteration: current optimization iteration idx

        output:
            extrinsic_w2c: initial extrinsic_w2c
            {"theta": theta, "rho": rho}: new updated delta_rot and delta_trans
        """
        if iteration < self.start_iter:
            theta = torch.zeros((1, 3), device=extrinsic_w2c.device, dtype=extrinsic_w2c.dtype)
            rho = torch.zeros((1, 3), device=extrinsic_w2c.device, dtype=extrinsic_w2c.dtype)
            return extrinsic_w2c, {"theta": theta, "rho": rho}

        if iteration <= self.end_iter:
            theta, rho = self.pose_updater(view_ids)
        else:
            with torch.no_grad():
                theta, rho = self.pose_updater(view_ids)
        theta, rho = self._align_pose_delta(theta, rho, extrinsic_w2c)
        refined_w2c = self._apply_delta_left_multiply(extrinsic_w2c, theta, rho)
        return refined_w2c, {"theta": theta, "rho": rho}

    def step(self, iteration: int):
        if self.start_iter <= iteration <= self.end_iter and self.update_steps > 0:
            self.rot_optimizer.step()
            self.trans_optimizer.step()
            if self.rot_scheduler is not None:
                self.rot_scheduler.step()
            if self.trans_scheduler is not None:
                self.trans_scheduler.step()
        self.rot_optimizer.zero_grad(set_to_none=True)
        self.trans_optimizer.zero_grad(set_to_none=True)

    def export_refined_train_w2c(self, train_dataset, device, scene_scale):
        refined_train_w2c = {}
        with torch.no_grad():
            for view_id, view_info in zip(train_dataset.view_ids, train_dataset.views_info_list):
                w2c = torch.tensor(view_info.extrinsic, dtype=torch.float32, device=device).unsqueeze(0)
                w2c[..., :3, 3] *= scene_scale
                theta, rho = self.pose_updater([int(view_id)])
                w2c_ref = self._apply_delta_left_multiply(w2c, theta, rho).squeeze(0)
                w2c_ref[:3, 3] /= scene_scale
                refined_train_w2c[int(view_id)] = w2c_ref.detach().cpu().numpy()
        return refined_train_w2c
