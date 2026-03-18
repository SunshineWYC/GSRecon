import os
import sys
import json
import time
import torch
import shutil
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from munch import munchify
from argparse import ArgumentParser
from utils.config_utils import load_config
from gaussian_splatting import create_renderer, create_pose_refiner
from utils.utils import create_dataloader, infinite_dataloader, load_pcdfile, collate_single_view
from utils.eval_utils import evaluate_gaussian_photometric
from torch.utils.tensorboard import SummaryWriter
from gaussian_splatting.gsplat.gaussian_model import GaussianModel
from utils.loss_utils import l1_loss
from utils.general_utils import build_scaling_rotation
from fused_ssim import fused_ssim as fast_ssim
from datasets.colmap_loader import COLMAPSceneInfo, COLMAPDataset
from datasets.colmap_reader import read_colmap_model, write_colmap_text


def _to_view_ids_list(view_id):
    """
    convert different dtype view_id to int list
    """
    if isinstance(view_id, torch.Tensor):
        return [int(v) for v in view_id.reshape(-1).tolist()]
    if isinstance(view_id, (list, tuple)):
        return [int(v) for v in view_id]
    return [int(view_id)]


def _safe_mean_metric(scores):
    if not scores:
        return 0.0
    return float(sum(scores.values()) / len(scores))


def optimize(train_dataset, eval_dataset, renderer, renderer_type, model_params, training_params, device, output_dirpath, pcd_filepath):
    t0 = time.perf_counter()
    # output dirpath pre setting
    log_dir = os.path.join(output_dirpath, "logs")
    os.makedirs(log_dir, exist_ok=True)
    gaussian_output_dir = os.path.join(output_dirpath, "gaussians")
    os.makedirs(gaussian_output_dir, exist_ok=True)

    # set up logger
    logger = SummaryWriter(log_dir=log_dir) if training_params.get("log", True) else None
    eval_interval = training_params.get("eval_interval", 1000)
    pose_optimize = training_params.get("pose_optimize", False)
    pose_refiner = None

    print("Num train views: ", len(train_dataset))
    dataloader = create_dataloader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        preload=training_params.get("preload", False),
        preload_device=training_params.get("preload_device", "cpu"),
    )
    data_iter = infinite_dataloader(dataloader)

    # gaussian model initialization
    gaussians = GaussianModel(sh_degree=model_params.sh_degree, device=device)
    pcd = load_pcdfile(pcd_filepath, scene_scale=model_params.get("scene_scale", 1.0))
    # Spatial unit scale for xyz learning-rate scaling and absgrad densify/prune thresholds.
    spatial_unit_scale = 5.0
    gaussians.create_from_pcd(pcd, spatial_unit_scale=spatial_unit_scale)
    gaussians.training_setup(training_params)

    gauss

    if pose_optimize:
        pose_updater_params = training_params.get("pose_updater_params", {})
        pose_optimizer_params = training_params.get("pose_optimizer_params", {})
        pose_refiner = create_pose_refiner(
            renderer_type=renderer_type,
            view_ids=train_dataset.view_ids,
            device=device,
            total_iters=training_params.iterations,
            pose_updater_params=pose_updater_params,
            pose_optimizer_params=pose_optimizer_params,
        )

    # optimization loop
    t1 = time.perf_counter()
    print("Start optimization, initial gaussian point number: {}".format(gaussians.get_xyz.shape[0]))
    iteration = 0

    pbar = tqdm(range(training_params.iterations), desc="Optimizing...", unit="it")
    for iter_idx in pbar:
        iteration = iter_idx + 1
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # update scheduler learning rates
        xyz_lr = gaussians.update_learning_rate(iteration)

        view_id, view_data = next(data_iter)
        
        extrinsic = view_data["extrinsic"].to(device, non_blocking=True)
        if pose_refiner is not None:
            view_ids_list = _to_view_ids_list(view_id)
            extrinsic = pose_refiner.refine_pose(extrinsic, view_ids_list, iteration)
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
            device=device
        )

        # Keep fp16 preload for storage, but normalize image math to fp32 for loss stability and kernel compatibility.
        image_rendered_f32 = image_rendered.float()
        image_gt_f32 = image_gt.float()

        # calculate loss
        loss_l1_color = l1_loss(image_rendered_f32, image_gt_f32)
        loss_ssim = 1.0 - fast_ssim(image_rendered_f32.unsqueeze(0), image_gt_f32.unsqueeze(0))
        loss = (1.0 - training_params.lambda_dssim) * loss_l1_color + training_params.lambda_dssim * loss_ssim

        # MCMC Regularization
        if training_params.densification_params.strategy == "mcmc":
            mcmc_cfg = training_params.densification_params.mcmc_params
            loss = loss + mcmc_cfg.get("opacity_reg", 0.01) * torch.abs(gaussians.get_opacity).mean()
            loss = loss + mcmc_cfg.get("scale_reg", 0.01) * torch.abs(gaussians.get_scaling).mean()

        loss.backward()

        with torch.no_grad():
            # Optimizer step for gaussians and pose_refiner
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            if pose_refiner is not None:    # pose_refiner step
                pose_refiner.step(iteration)

            # adaptive density control: densification and prune
            if training_params.densification_params.strategy == "absgrad":
                # Absgrad densification strategy
                if iteration < training_params.densification_params.densify_until_iter:  # accumulate stats
                    visible_gaussian_ids = info["gaussian_ids"]
                    max_radii2D = info["radii"].max(dim=1).values
                    gaussians.max_radii2D[visible_gaussian_ids] = torch.max(gaussians.max_radii2D[visible_gaussian_ids], max_radii2D)
                    gaussians.add_densification_stats_packed(info)
                # Absgrad: perform densification, pruning and reset opacity
                if training_params.densification_params.densify_from_iter < iteration < training_params.densification_params.densify_until_iter:
                    if iteration % training_params.densification_params.densification_interval == 0:
                        size_threshold = 20 if iteration > training_params.densification_params.opacity_reset_interval else None
                        gaussians.densify_and_prune(
                            training_params.densification_params.absgrad_params.densify_grad_threshold,
                            training_params.densification_params.min_opacity,
                            spatial_unit_scale,
                            size_threshold,)
                    if iteration % training_params.densification_params.opacity_reset_interval == 0:
                        gaussians.reset_opacity()
            elif training_params.densification_params.strategy == "mcmc":
                # MCMC densification
                if (iteration < training_params.densification_params.densify_until_iter
                    and iteration > training_params.densification_params.densify_from_iter
                    and iteration % training_params.densification_params.densification_interval == 0
                ):
                    dead_mask = (gaussians.get_opacity <= training_params.densification_params.min_opacity).squeeze(-1)
                    gaussians.relocate_gs(dead_mask=dead_mask)
                    gaussians.add_new_gs(
                        cap_max=model_params.max_num_gaussians, 
                        new_gaussian_ratio=training_params.densification_params.mcmc_params.new_gaussian_ratio)
            else:
                raise ValueError(f"Unsupported densification strategy: {training_params.densification_params.strategy}")

            # MCMC noise SGLD addition
            if training_params.densification_params.strategy == "mcmc":
                def op_sigmoid(x, k=100, x0=0.995):
                    return 1 / (1 + torch.exp(-k * (x - x0)))
                
                L = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
                actual_covariance = L @ L.transpose(1, 2)
                
                # Generate noise based on the current point positions
                noise = (
                    torch.randn_like(gaussians._xyz)
                    * op_sigmoid(1 - gaussians.get_opacity)
                    * float(training_params.densification_params.mcmc_params.noise_lr)
                    * xyz_lr
                )
                # Scale noise by the local covariance (SGLD)
                noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
                gaussians._xyz.add_(noise)

        # logging
        if logger:
            if iteration % training_params.log_interval == 0:
                logger.add_scalar("Loss/loss_l1_color", loss_l1_color.item(), iteration)
                logger.add_scalar("Loss/loss_ssim", loss_ssim.item(), iteration)
                logger.add_scalar("Loss/loss", loss.item(), iteration)
                logger.add_scalar("Stats/num_gaussians", gaussians.get_xyz.shape[0], iteration)
            
            # training image logging
            # if iteration % training_params.image_log_interval == 0:
            #     logger.add_image("image/image_rendered", image_rendered, iteration)
            #     logger.add_image("image/image_gt", image_gt, iteration)

            # evaluation on dataset and metrics logging
            if training_params.get("eval", False) and (iteration == 1 or iteration % eval_interval == 0):
                # Evaluate on train split with refined pose (if pose refinement is enabled).
                train_psnr_scores, train_ssim_scores, train_lpips_scores = evaluate_gaussian_photometric(
                    gaussians,
                    train_dataset,
                    renderer,
                    lpips_flag=True,
                    device=device,
                    pose_refiner=pose_refiner,
                    pose_iteration=iteration,
                )
                logger.add_scalar("Metrics/train_psnr", _safe_mean_metric(train_psnr_scores), iteration)
                logger.add_scalar("Metrics/train_ssim", _safe_mean_metric(train_ssim_scores), iteration)
                logger.add_scalar("Metrics/train_lpips", _safe_mean_metric(train_lpips_scores), iteration)

                # Evaluate on val split with original poses.
                eval_psnr_scores, eval_ssim_scores, eval_lpips_scores = evaluate_gaussian_photometric(
                    gaussians,
                    eval_dataset,
                    renderer,
                    lpips_flag=True,
                    device=device,
                )
                logger.add_scalar("Metrics/eval_psnr", _safe_mean_metric(eval_psnr_scores), iteration)
                logger.add_scalar("Metrics/eval_ssim", _safe_mean_metric(eval_ssim_scores), iteration)
                logger.add_scalar("Metrics/eval_lpips", _safe_mean_metric(eval_lpips_scores), iteration)

        pbar.set_postfix(loss=f"{loss.item():.4f}", num_gs=f"{gaussians.get_xyz.shape[0]}")

    # save gaussian model
    t2 = time.perf_counter()
    gaussians.save_ply(os.path.join(gaussian_output_dir, "gaussian_{}.ply".format(iteration)))
    print("End optimization, final gaussian point number : {}.".format(gaussians.get_xyz.shape[0]))

    # final evaluation
    t3 = time.perf_counter()
    timing = {
        "data_load": t1 - t0,
        "optimization": t2 - t1,
        "save_model": t3 - t2,
        "total": t3 - t0,
    }
    with open(os.path.join(log_dir, "timing.json"), "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(
        "Timing: data_load={:.3f}s, optimization={:.3f}s, save_model={:.3f}s, total={:.3f}s".format(
            timing["data_load"],
            timing["optimization"],
            timing["save_model"],
            timing["total"],
        )
    )
    
    # update poses with pose_refine, write as COLMAP format
    refined_train_w2c = None
    if pose_refiner is not None:
        scene_scale = float(model_params.get("scene_scale", 1.0))
        refined_train_w2c = pose_refiner.export_refined_train_w2c(train_dataset, device, scene_scale)

    return refined_train_w2c


if __name__ == "__main__":
    parser = ArgumentParser(description="Spatial block training script parameters")
    parser.add_argument("--config", type=str, default="configs/gsplat/truck.yaml", help="Path to the configuration file")
    args = parser.parse_args(sys.argv[1:])
    config = load_config(args.config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_root = os.path.join(config["scene"]["output_path"], config["scene"]["exp_name"])
    os.makedirs(output_root, exist_ok=True)
    shutil.copy(args.config, os.path.join(output_root, "config.yaml"))

    model_params = munchify(config.get("model_params", {}))
    training_params = munchify(config.get("training_params", {}))
    renderer_params = munchify(config.get("renderer_params", {}))

    # renderer definition
    renderer_cfg = dict(renderer_params)
    renderer_type = renderer_cfg.pop("renderer_type", "gsplat")
    renderer = create_renderer(renderer_type=renderer_type, **renderer_cfg)

    # scene info and pcd filepath
    scene_info = COLMAPSceneInfo(config["scene"]["data_path"])
    pcd_filepath = scene_info.pcd_filepath
    train_split = "train" if bool(training_params.get("eval", False)) else None
    train_dataset = COLMAPDataset(
        views_info=scene_info.views_info,
        image_scale=model_params.get("image_scale", 1.0),
        scene_scale=model_params.get("scene_scale", 1.0),
        preload=training_params.get("preload", False),
        preload_device=training_params.get("preload_device", "cpu"),
        preload_dtype=training_params.get("preload_dtype", "fp32"),
        split=train_split,
        eval_split_interval=training_params.get("eval_split_interval", 8),
    )
    if bool(training_params.get("eval", False)):
        eval_dataset = COLMAPDataset(
            views_info=scene_info.views_info,
            image_scale=model_params.get("image_scale", 1.0),
            scene_scale=model_params.get("scene_scale", 1.0),
            preload=training_params.get("preload", False),
            preload_device=training_params.get("preload_device", "cpu"),
            preload_dtype=training_params.get("preload_dtype", "fp32"),
            split="val",
            eval_split_interval=training_params.get("eval_split_interval", 8),
        )
    else:
        eval_dataset = None

    refined_train_w2c = optimize(
        train_dataset,
        eval_dataset,
        renderer=renderer,
        renderer_type=renderer_type,
        model_params=model_params,
        training_params=training_params,
        device=device,
        output_dirpath=output_root,
        pcd_filepath=pcd_filepath
    )
    
    # save refined poses
    if training_params.get("pose_optimize", False) and refined_train_w2c:
        images, cameras = read_colmap_model(config["scene"]["data_path"])
        output_sparse_dir = os.path.join(output_root, "pose_refined", "sparse", "0")
        write_colmap_text(output_sparse_dir, images, cameras, refined_train_w2c)

    print("Optimization Finished.")
