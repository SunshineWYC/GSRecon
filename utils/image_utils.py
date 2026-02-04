#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import PIL
import PIL.Image
import torch
import numpy as np


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    # mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    mse = (img1 - img2).pow(2).mean()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def psnr_mask(img1, img2, mask):
    """
    Masked PSNR: only pixels with mask>0 contribute to MSE.
    img1, img2: [B, C, H, W]; mask: [H, W], [B, H, W], or [B, 1, H, W]
    """
    if mask.dim() == 2:       # [H, W] -> [1, H, W]
        mask = mask.unsqueeze(0)
    if mask.dim() == 4:       # [B,1,H,W] -> [B,H,W]
        mask = mask[:, 0, :, :]
    mask = (mask > 0).float() # [B, H, W]

    B, C, H, W = img1.shape
    mask_flat = mask.reshape(B, -1)                  # [B, HW]
    mask_flat = mask_flat.unsqueeze(1).expand(-1, C, -1)  # [B, C, HW]

    img1_flat = img1.reshape(B, C, -1)
    img2_flat = img2.reshape(B, C, -1)
    diff2 = (img1_flat - img2_flat).pow(2)

    denom = mask_flat.sum(dim=2).clamp_min(1.0)      # [B, C]
    mse_per_channel = (diff2 * mask_flat).sum(dim=2) / denom
    mse = mse_per_channel.mean(dim=1, keepdim=True)  # average over channels
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def read_image(filename, image_scale=1.0):
    image = PIL.Image.open(filename)
    new_width = int(image.width * image_scale)
    new_height = int(image.height * image_scale)
    image = image.resize((new_width, new_height))
    np_image = np.array(image, dtype=np.float32) / 255.0

    return np_image[:, :, :3]


def save_image(image:np.ndarray, filepath):
    image_save = PIL.Image.fromarray((image*255).astype("uint8"))
    image_save.save(filepath)
