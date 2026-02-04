import json
import os
import re
import sys
from argparse import ArgumentParser

import torch

from datasets.colmap_loader import COLMAPSceneInfo, COLMAPDataset
from gaussian_splatting.gaussian_model import GaussianModel
from renderer import create_renderer
from utils.config_utils import load_config
from utils.eval_utils import evaluate_gaussian_photometric


def _select_gaussian_ply(gaussians_dirpath: str) -> str:
    if not os.path.isdir(gaussians_dirpath):
        raise FileNotFoundError(f"gaussians directory not found: {gaussians_dirpath}")

    ply_filenames = [fn for fn in os.listdir(gaussians_dirpath) if fn.endswith(".ply")]
    if not ply_filenames:
        raise FileNotFoundError(f"No .ply files found under: {gaussians_dirpath}")

    iter_re = re.compile(r"^iteration_(\d+)\.ply$")
    candidates = []
    for fn in ply_filenames:
        m = iter_re.match(fn)
        if m:
            candidates.append((int(m.group(1)), fn))

    if candidates:
        _, best_fn = max(candidates, key=lambda x: x[0])
        return os.path.join(gaussians_dirpath, best_fn)

    # Fallback: latest modified .ply
    best_fn = max(ply_filenames, key=lambda fn: os.path.getmtime(os.path.join(gaussians_dirpath, fn)))
    return os.path.join(gaussians_dirpath, best_fn)


def _avg(values):
    values = list(values)
    if not values:
        return None
    return float(sum(values) / len(values))


def _build_payload(dataset, psnr_scores, ssim_scores, lpips_scores):
    frames = []
    for view_id, view_info in zip(dataset.view_ids, dataset.views_info_list):
        filename = os.path.basename(view_info.image_filepath)
        frames.append(
            {
                "filename": filename,
                "psnr": psnr_scores[view_id],
                "ssim": ssim_scores[view_id],
                "lpips": lpips_scores[view_id],
            }
        )

    return {
        "avg_psnr": _avg(psnr_scores.values()),
        "avg_ssim": _avg(ssim_scores.values()),
        "avg_lpips": _avg(lpips_scores.values()),
        "frames": frames,
    }


def evaluate_photometric(exp_dir, config, gaussians, device, splits):
    scene_cfg = config.get("scene", {})
    training_params = config.get("training_params", {})
    renderer_params = config.get("renderer_params", {})
    model_params = config.get("model_params", {})

    data_path = scene_cfg.get("data_path")
    if not data_path:
        raise ValueError("Missing `scene.data_path` in config.yaml")

    image_scale = float(model_params.get("image_scale", 1.0))
    scene_scale = float(model_params.get("scene_scale", 1.0))
    preload = bool(training_params.get("preload", False))
    eval_split_interval = int(training_params.get("eval_split_interval", 8))

    renderer_cfg = dict(renderer_params)
    renderer_type = renderer_cfg.pop("renderer_type", "gsplat")
    renderer = create_renderer(renderer_type=renderer_type, **renderer_cfg)

    scene_info = COLMAPSceneInfo(data_path)
    metrics_dir = os.path.join(exp_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    if "train" in splits:
        train_dataset = COLMAPDataset(
            views_info=scene_info.views_info,
            image_scale=image_scale,
            scene_scale=scene_scale,
            preload=preload,
            split="train",
            eval_split_interval=eval_split_interval,
        )
        train_psnr, train_ssim, train_lpips = evaluate_gaussian_photometric(
            gaussians, train_dataset, renderer, lpips_flag=True, device=device
        )
        train_payload = _build_payload(train_dataset, train_psnr, train_ssim, train_lpips)
        with open(os.path.join(metrics_dir, "train.json"), "w", encoding="utf-8") as f:
            json.dump(train_payload, f, indent=2, ensure_ascii=False)
        print(
            f"TRAIN avg_psnr={train_payload['avg_psnr']}, "
            f"avg_ssim={train_payload['avg_ssim']}, "
            f"avg_lpips={train_payload['avg_lpips']}"
        )
        print(f"Saved: {os.path.join(metrics_dir, 'train.json')}")

    if "val" in splits:
        val_dataset = COLMAPDataset(
            views_info=scene_info.views_info,
            image_scale=image_scale,
            scene_scale=scene_scale,
            preload=preload,
            split="val",
            eval_split_interval=eval_split_interval,
        )
        val_psnr, val_ssim, val_lpips = evaluate_gaussian_photometric(
            gaussians, val_dataset, renderer, lpips_flag=True, device=device
        )
        val_payload = _build_payload(val_dataset, val_psnr, val_ssim, val_lpips)
        with open(os.path.join(metrics_dir, "val.json"), "w", encoding="utf-8") as f:
            json.dump(val_payload, f, indent=2, ensure_ascii=False)
        print(
            f"VAL avg_psnr={val_payload['avg_psnr']}, "
            f"avg_ssim={val_payload['avg_ssim']}, "
            f"avg_lpips={val_payload['avg_lpips']}"
        )
        print(f"Saved: {os.path.join(metrics_dir, 'val.json')}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a Gaussian reconstruction for a single experiment.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory, e.g. results/truck/exp_001")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device string, e.g. cuda:0")
    parser.add_argument("--ply", type=str, default=None, help="Optional .ply path; default selects max iteration under gaussians/")
    parser.add_argument(
        "--split",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
        help='Splits to evaluate, e.g. --split train val or --split train',
    )
    args = parser.parse_args()

    exp_dir = args.exp_dir
    config_path = os.path.join(exp_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.yaml not found: {config_path}")

    gaussians_dir = os.path.join(exp_dir, "gaussians")
    ply_path = args.ply or _select_gaussian_ply(gaussians_dir)

    config = load_config(config_path)
    model_params = config.get("model_params", {})

    device = torch.device(args.device)

    spherical_harmonics = bool(model_params.get("spherical_harmonics", True))
    sh_degree_cfg = int(model_params.get("sh_degree", 0))
    sh_degree = sh_degree_cfg if spherical_harmonics else 0
    gaussians = GaussianModel(sh_degree=sh_degree, device=device)
    gaussians.load_ply(ply_path)

    evaluate_photometric(exp_dir, config, gaussians, device, args.split)

    # Other evaluation (TODO)
