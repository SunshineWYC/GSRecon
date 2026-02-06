from renderer.gsplat_renderer import GSplatRenderer, GSplatPoseRefiner


def create_renderer(renderer_type, **kwargs):
    if renderer_type == "gsplat":
        return GSplatRenderer(**kwargs)
    else:
        return GSplatRenderer(**kwargs)


def create_pose_refiner(renderer_type, view_ids, device, total_iters, pose_updater_params, pose_optimizer_params):
    if renderer_type == "gsplat":
        return GSplatPoseRefiner(
            view_ids=view_ids,
            device=device,
            total_iters=total_iters,
            pose_updater_params=pose_updater_params,
            pose_optimizer_params=pose_optimizer_params,
        )
    else:
        return GSplatPoseRefiner(
            view_ids=view_ids,
            device=device,
            total_iters=total_iters,
            pose_updater_params=pose_updater_params,
            pose_optimizer_params=pose_optimizer_params,
        )
