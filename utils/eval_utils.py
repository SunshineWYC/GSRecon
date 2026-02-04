import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.image_utils import psnr
from utils.lpipsPyTorch.modules.lpips import LPIPS
from gsplat.rendering import rasterization
from gaussian_splatting.utils.loss_utils import create_window
from fused_ssim import fused_ssim as fast_ssim


def ssim_mask(img1, img2, mask, window_size=11):
    """
    Masked SSIM: only pixels with mask>0 contribute to the average.
    img1, img2: [B, C, H, W], mask: [B, 1, H, W] or [B, H, W]
    """
    if mask.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
        mask = mask.unsqueeze(1)
    mask = (mask > 0).float()

    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # Compute per-pixel SSIM map (adapted from loss_utils._ssim without averaging)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    weight = mask.expand_as(ssim_map)
    denom = weight.sum().clamp_min(1.0)
    return (ssim_map * weight).sum() / denom


@torch.no_grad()
def evaluate_gaussian_model(gaussians, dataset, renderer, lpips_flag=True, device="cuda:0"):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    dataloader_iter = iter(dataloader)

    psnr_scores, ssim_scores, lpips_scores = {}, {}, {}
    lpips_model = None
    if lpips_flag:
        lpips_model = LPIPS(net_type="vgg", version="0.1").to(device)
        lpips_model.eval()

    for idx in range(len(dataset)):
        view_id, view_data = next(dataloader_iter)
        extrinsic = view_data["extrinsic"].to(device, non_blocking=True)
        intrinsic = view_data["intrinsic"].to(device, non_blocking=True)
        image_height, image_width = view_data["height"], view_data["width"]
        image_gt = view_data["image"][0].to(device, non_blocking=True)  # [3, H, W]

        image_rendered, depth_rendered, info = renderer.render(
            gaussians=gaussians,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            sh_degree=gaussians.active_sh_degree,
            image_height=image_height,
            image_width=image_width,
        )

        psnr_score = psnr(image_rendered, image_gt)
        ssim_score = fast_ssim(image_rendered.unsqueeze(0), image_gt.unsqueeze(0))
        if lpips_flag:
            lpips_score = lpips_model(image_rendered.unsqueeze(0), image_gt.unsqueeze(0))
        else:
            lpips_score = torch.tensor(0.0)

        psnr_scores[view_id] = psnr_score.item()
        ssim_scores[view_id] = ssim_score.item()
        lpips_scores[view_id] = lpips_score.item()

    return psnr_scores, ssim_scores, lpips_scores
