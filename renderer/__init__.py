from renderer.gsplat_renderer import GSplatRenderer, GSplatPoseRefiner
from renderer.fastgs_renderer import FastGSRenderer, FastGSPoseRefiner


def create_renderer(renderer_type, **kwargs):
    if renderer_type == "gsplat":
        return GSplatRenderer(**kwargs)
    elif renderer_type == "fastgs":
        return FastGSRenderer(**kwargs)
    else:
        raise ValueError(f"Unknown renderer_type: {renderer_type}")


def create_pose_refiner(renderer_type, view_ids, device, total_iters, pose_updater_params, pose_optimizer_params):
    if renderer_type == "gsplat":
        return GSplatPoseRefiner(
            view_ids=view_ids,
            device=device,
            total_iters=total_iters,
            pose_updater_params=pose_updater_params,
            pose_optimizer_params=pose_optimizer_params,
        )
    elif renderer_type == "fastgs":
        return FastGSPoseRefiner(
            view_ids=view_ids,
            device=device,
            total_iters=total_iters,
            pose_updater_params=pose_updater_params,
            pose_optimizer_params=pose_optimizer_params,
        )
    else:
        raise ValueError(f"Unknown renderer_type: {renderer_type}")
