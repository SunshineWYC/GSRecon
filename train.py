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
from utils.eval_utils import evaluate_gaussian_photometric
from torch.utils.tensorboard import SummaryWriter
from gaussian_splatting.gaussian_model import GaussianModel
from gaussian_splatting.pose_optimization import CameraOptModule
from gaussian_splatting.utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from datasets.colmap_loader import COLMAPSceneInfo, COLMAPDataset
from datasets.colmap_reader import (
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_text,
    rotmat2qvec,
)


def _read_colmap_model(scene_dirpath):
    sparse_dir = os.path.join(scene_dirpath, "sparse/0")
    try:
        cameras_extrinsic_file = os.path.join(sparse_dir, "images.bin")
        cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.bin")
        images = read_extrinsics_binary(cameras_extrinsic_file)
        cameras = read_intrinsics_binary(cameras_intrinsic_file)
    except Exception:
        cameras_extrinsic_file = os.path.join(sparse_dir, "images.txt")
        cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.txt")
        images = read_extrinsics_text(cameras_extrinsic_file)
        cameras = read_intrinsics_text(cameras_intrinsic_file)
    return images, cameras


def _write_colmap_text(output_sparse_dir, images, cameras, refined_train_w2c):
    os.makedirs(output_sparse_dir, exist_ok=True)

    cameras_path = os.path.join(output_sparse_dir, "cameras.txt")
    with open(cameras_path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for camera_id in sorted(cameras.keys()):
            cam = cameras[camera_id]
            params = " ".join(str(p) for p in cam.params)
            f.write(f"{cam.id} {cam.model} {cam.width} {cam.height} {params}\n")

    images_path = os.path.join(output_sparse_dir, "images.txt")
    with open(images_path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image_id in sorted(images.keys()):
            img = images[image_id]
            if image_id in refined_train_w2c:
                w2c = refined_train_w2c[image_id]
                qvec = rotmat2qvec(w2c[:3, :3])
                tvec = w2c[:3, 3]
            else:
                qvec = img.qvec
                tvec = img.tvec
            f.write(
                f"{image_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                f"{tvec[0]} {tvec[1]} {tvec[2]} {img.camera_id} {img.name}\n"
            )
            if img.xys is not None and len(img.xys) > 0:
                pts = []
                for (x, y), pid in zip(img.xys, img.point3D_ids):
                    pts.append(f"{x} {y} {pid}")
                f.write(" ".join(pts) + "\n")
            else:
                f.write("\n")

    points_path = os.path.join(output_sparse_dir, "points3D.txt")
    with open(points_path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")


def optimize(train_dataset, eval_dataset, renderer, model_params, training_params, device, output_dirpath, pcd_filepath):
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
    pose_params = training_params.get("pose_optim_params", {})
    pose_refine = None
    pose_optimizer = None
    pose_scheduler = None
    pose_start = int(pose_params.get("start_iter", 2000))
    pose_end = int(pose_params.get("end_iter", 25000))

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
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )
    data_iter = infinite_dataloader(dataloader)

    # gaussian model initialization
    gaussians = GaussianModel(sh_degree=model_params.sh_degree, device=device)
    pcd = load_pcdfile(pcd_filepath, scene_scale=model_params.get("scene_scale", 1.0))
    scene_extent = 5.0
    gaussians.create_from_pcd(pcd, scene_extent=scene_extent)
    gaussians.training_setup(training_params)

    if pose_optimize:
        train_view_ids = train_dataset.view_ids
        pose_refine = CameraOptModule(train_view_ids).to(device)
        pose_refine.zero_init()
        pose_lr = float(pose_params.get("lr", 1e-5))
        pose_reg = float(pose_params.get("reg", 1e-6))
        pose_optimizer = torch.optim.Adam(pose_refine.parameters(), lr=pose_lr, weight_decay=pose_reg)
        gamma_final_ratio = float(pose_params.get("gamma_final_ratio", 0.01))
        pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            pose_optimizer, gamma=gamma_final_ratio ** (1.0 / training_params.iterations)
        )

    # optimization loop
    t1 = time.perf_counter()
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
        if pose_refine is not None:
            if isinstance(view_id, torch.Tensor):
                view_ids_list = [int(view_id.reshape(-1)[0].item())]
            elif isinstance(view_id, (list, tuple)):
                view_ids_list = [int(v) for v in view_id]
            else:
                view_ids_list = [int(view_id)]
            camtoworld = torch.linalg.inv(extrinsic)
            camtoworld = pose_refine(camtoworld, view_ids_list)
            extrinsic = torch.linalg.inv(camtoworld)
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

        # calculate loss
        loss_l1_color = l1_loss(image_rendered, image_gt)
        loss_ssim = 1.0 - fast_ssim(image_rendered.unsqueeze(0), image_gt.unsqueeze(0))
        loss_color = (1.0 - training_params.lambda_dssim) * loss_l1_color + training_params.lambda_dssim * loss_ssim

        loss = loss_color
        loss.backward()

        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            if pose_refine is not None:
                if pose_start <= iteration <= pose_end:
                    pose_optimizer.step()
                    pose_scheduler.step()
                pose_optimizer.zero_grad(set_to_none=True)

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
                    psnr_scores, ssim_scores, lpips_scores = evaluate_gaussian_photometric(gaussians, eval_dataset, renderer, lpips_flag=True, device=device)
                    logger.add_scalar("Metrics/eval_psnr", sum(psnr_scores.values()) / len(psnr_scores), iteration)
                    logger.add_scalar("Metrics/eval_ssim", sum(ssim_scores.values()) / len(ssim_scores), iteration)
                    logger.add_scalar("Metrics/eval_lpips", sum(lpips_scores.values()) / len(lpips_scores), iteration)

        pbar.set_postfix(loss=f"{loss.item():.4f}", num_gs=f"{gaussians.get_xyz.shape[0]}")

    # save gaussian model
    t2 = time.perf_counter()
    gaussians.save_ply(os.path.join(gaussian_output_dir, "iteration_{}.ply".format(iteration)))
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
    
    refined_train_w2c = None
    if pose_refine is not None:
        refined_train_w2c = {}
        scene_scale = float(model_params.get("scene_scale", 1.0))
        with torch.no_grad():
            for view_id, view_info in zip(train_dataset.view_ids, train_dataset.views_info_list):
                w2c = torch.tensor(view_info.extrinsic, dtype=torch.float32, device=device).unsqueeze(0)
                w2c[..., :3, 3] *= scene_scale
                camtoworld = torch.linalg.inv(w2c)
                camtoworld = pose_refine(camtoworld, [int(view_id)])
                w2c_ref = torch.linalg.inv(camtoworld).squeeze(0)
                w2c_ref[:3, 3] /= scene_scale
                refined_train_w2c[int(view_id)] = w2c_ref.detach().cpu().numpy()

    return refined_train_w2c


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
        eval_split_interval=training_params.get("eval_split_interval", 8),
    )
    if training_params.get("eval", False):
        eval_dataset = COLMAPDataset(
            views_info=scene_info.views_info,
            image_scale=model_params.get("image_scale", 1.0),
            scene_scale=model_params.get("scene_scale", 1.0),
            preload=training_params.get("preload", False),
            split="val",
            eval_split_interval=training_params.get("eval_split_interval", 8),
        )
    else:
        eval_dataset = None

    refined_train_w2c = optimize(
        train_dataset,
        eval_dataset,
        renderer=renderer,
        model_params=model_params,
        training_params=training_params,
        device=device,
        output_dirpath=output_root,
        pcd_filepath=pcd_filepath
    )

    if training_params.get("pose_optimize", False) and refined_train_w2c:
        images, cameras = _read_colmap_model(config["scene"]["data_path"])
        output_sparse_dir = os.path.join(output_root, "pose_refined", "sparse", "0")
        _write_colmap_text(output_sparse_dir, images, cameras, refined_train_w2c)

    print("Hello World!")
