import os
import gsplat



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
