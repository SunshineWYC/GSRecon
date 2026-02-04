import os
import sys
import json
import time
import torch
import shutil
import itertools
import numpy as np
from tqdm import tqdm
from munch import munchify
from argparse import ArgumentParser
from utils.config_utils import load_config
from renderer import create_renderer
from utils.utils import infinite_dataloader, load_pcdfile
from utils.eval_utils import evaluate_gaussian_model
from torch.utils.tensorboard import SummaryWriter
from gaussian_splatting.gaussian_model import GaussianModel
from gaussian_splatting.utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from datasets.colmap_loader import COLMAPSceneInfo, COLMAPDataset


def optimization(train_dataset, eval_dataset, renderer, model_params, training_params, device, output_dirpath, pcd_filepath):
    # output dirpath pre setting
    log_dir = os.path.join(output_dirpath, "logs")
    os.makedirs(log_dir, exist_ok=True)
    gaussian_output_dir = os.path.join(output_dirpath, "gaussians")
    os.makedirs(gaussian_output_dir, exist_ok=True)

    # set up logger
    logger = SummaryWriter(log_dir=log_dir) if training_params.get("log", True) else None
    eval_interval = training_params.get("eval_interval", 1000)

    # dataloader definition
    if training_params.get("preload", False):
        dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=8,
            drop_last=False,
        )
    data_iter = infinite_dataloader(dataloader)

    # gaussian model initialization
    gaussians = GaussianModel(sh_degree=model_params.sh_degree)
    pcd = load_pcdfile(pcd_filepath, scene_scale=model_params.get("scene_scale", 1.0))
    scene_extent = 5.0
    gaussians.create_from_pcd(pcd, scene_extent=scene_extent)
    gaussians.training_setup(training_params)

    # optimization loop
    print("Start optimization, initial gaussian point number: {}".format(gaussians.get_xyz.shape[0]))
    iteration = 0

    pbar = tqdm(range(training_params.iterations), desc="Optimizing...", unit="it")
    for iter_idx in pbar:
        iteration = iter_idx + 1
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        view_id, view_data = next(data_iter)
        
        extrinsic = view_data["extrinsic"].to(device, non_blocking=True)
        intrinsic = view_data["intrinsic"].to(device, non_blocking=True)
        image_height, image_width = view_data["height"], view_data["width"]
        image_gt = view_data["image"][0].to(device, non_blocking=True)

        image_rendered, depth_rendered, info = renderer.render(
            gaussians=gaussians,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            sh_degree=gaussians.active_sh_degree,
            image_height=image_height,
            image_width=image_width,
        )

        # calculate loss
        loss_l1_color = l1_loss(image_rendered, image_gt)
        loss_ssim = 1.0 - fast_ssim(image_rendered.unsqueeze(0), image_gt.unsqueeze(0))
        loss_color = (1.0 - training_params.lambda_dssim) * loss_l1_color + training_params.lambda_dssim * loss_ssim

        loss = loss_color
        loss.backward()

        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            # adaptive density control: densification and prune
            # accumulate densification stats
            if iteration < training_params.densify_until_iter:
                visible_gaussian_ids = info["gaussian_ids"]
                max_radii2D = info["radii"].max(dim=1).values
                gaussians.max_radii2D[visible_gaussian_ids] = torch.max(gaussians.max_radii2D[visible_gaussian_ids], max_radii2D)
                gaussians.add_densification_stats_packed(info)

            # perform densification, pruning and reset opacity
            if training_params.densify_from_iter < iteration < training_params.densify_until_iter:
                if iteration % training_params.densification_interval == 0:
                    size_threshold = 20 if iteration > training_params.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        training_params.densify_grad_threshold,
                        0.005,
                        scene_extent,
                        size_threshold,
                        model_params.max_num_gaussians,
                    )

                if iteration % training_params.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

        # logging
        if logger:
            if iteration % training_params.log_interval == 0:
                logger.add_scalar("Loss/loss_l1_color", loss_l1_color.item(), iteration)
                logger.add_scalar("Loss/loss_ssim", loss_ssim.item(), iteration)
                logger.add_scalar("Loss/loss_color", loss_color.item(), iteration)
                logger.add_scalar("Loss/loss", loss.item(), iteration)
                logger.add_scalar("Stats/num_gaussians", gaussians.get_xyz.shape[0], iteration)
            
            # training image logging
            if iteration % training_params.image_log_interval == 0:
                logger.add_image("image/image_rendered", image_rendered, iteration)
                logger.add_image("image/image_gt", image_gt, iteration)

            # evaluation on dataset and metrics logging
            if training_params.get("eval", False):
                if iteration == 1 or iteration % eval_interval == 0:
                    psnr_scores, ssim_scores, lpips_scores = evaluate_gaussian_model(gaussians, eval_dataset, renderer, lpips_flag=True, device=device)
                    logger.add_scalar("Metrics/eval_psnr", sum(psnr_scores.values()) / len(psnr_scores), iteration)
                    logger.add_scalar("Metrics/eval_ssim", sum(ssim_scores.values()) / len(ssim_scores), iteration)
                    logger.add_scalar("Metrics/eval_lpips", sum(lpips_scores.values()) / len(lpips_scores), iteration)

        pbar.set_postfix(loss=f"{loss.item():.4f}", num_gs=f"{gaussians.get_xyz.shape[0]}")

    # save gaussian model
    gaussians.save_ply(os.path.join(gaussian_output_dir, "iteration_{}.ply".format(iteration)))
    print("End optimization, final gaussian point number : {}.".format(gaussians.get_xyz.shape[0]))

    # final evaluation
    psnr_scores, ssim_scores, lpips_scores = evaluate_gaussian_model(gaussians, train_dataset, renderer, lpips_flag=True, device=device)
    avg_psnr = sum(psnr_scores.values()) / len(psnr_scores)
    avg_ssim = sum(ssim_scores.values()) / len(ssim_scores)
    avg_lpips = sum(lpips_scores.values()) / len(lpips_scores)
    print("Final Evaluation on Train Views - PSNR: {:.4f}, SSIM: {:.4f}, LPIPS: {:.4f}".format(avg_psnr, avg_ssim, avg_lpips))

    if training_params.get("eval", False):
        psnr_scores, ssim_scores, lpips_scores = evaluate_gaussian_model(gaussians, eval_dataset, renderer, lpips_flag=True, device=device)
        avg_psnr = sum(psnr_scores.values()) / len(psnr_scores)
        avg_ssim = sum(ssim_scores.values()) / len(ssim_scores)
        avg_lpips = sum(lpips_scores.values()) / len(lpips_scores)
        print("Final Evaluation on Eval Views - PSNR: {:.4f}, SSIM: {:.4f}, LPIPS: {:.4f}".format(avg_psnr, avg_ssim, avg_lpips))


if __name__ == "__main__":
    parser = ArgumentParser(description="Spatial block training script parameters")
    parser.add_argument("--config", type=str, default="configs/truck.yaml", help="Path to the configuration file")
    args = parser.parse_args(sys.argv[1:])
    config = load_config(args.config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_root = os.path.join(config["scene"]["output_path"], config["scene"]["exp_name"])
    os.makedirs(output_root, exist_ok=True)
    shutil.copy(args.config, os.path.join(output_root, "config.yaml"))

    model_params = munchify(config.get("model_params", {}))
    training_params = munchify(config.get("training_params", {}))
    renderer_params = munchify(config.get("renderer_params", {}))
    model_params.sh_degree = model_params.get("sh_degree", 0) if model_params.spherical_harmonics else 0

    # renderer definition
    renderer = create_renderer(renderer_type=renderer_params.get("renderer_type", "gsplat"))

    # scene info and pcd filepath
    scene_info = COLMAPSceneInfo(config["scene"]["data_path"])
    pcd_filepath = scene_info.pcd_filepath
    train_dataset = COLMAPDataset(
        views_info=scene_info.views_info,
        image_scale=model_params.get("image_scale", 1.0),
        scene_scale=model_params.get("scene_scale", 1.0),
        preload=training_params.get("preload", False),
        split="train",
    )
    if training_params.get("eval", False):
        eval_dataset = COLMAPDataset(
            views_info=scene_info.views_info,
            image_scale=model_params.get("image_scale", 1.0),
            scene_scale=model_params.get("scene_scale", 1.0),
            preload=training_params.get("preload", False),
            split="val",
        )
    else:
        eval_dataset = None

    optimization(
        train_dataset,
        eval_dataset,
        renderer=renderer,
        model_params=model_params,
        training_params=training_params,
        device=device,
        output_dirpath=output_root,
        pcd_filepath=pcd_filepath
    )


    print("Hello World!")
