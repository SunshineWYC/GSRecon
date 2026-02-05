import torch
from torch import Tensor
import torch.nn.functional as F


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, view_ids):
        super().__init__()
        self.view_id_to_embed = {int(view_id): idx for idx, view_id in enumerate(view_ids)}
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(len(view_ids), 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def forward(self, camtoworlds: Tensor, view_ids) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            view_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        view_ids_list = [int(v) for v in view_ids]
        embed_list = []
        for vid in view_ids_list:
            if vid not in self.view_id_to_embed:
                raise KeyError(f"view_id {vid} is not in CameraOptModule mapping.")
            embed_list.append(self.view_id_to_embed[vid])

        embed_indices = torch.tensor(embed_list, device=camtoworlds.device, dtype=torch.long)
        assert camtoworlds.shape[:-2] == embed_indices.shape
        batch_dims = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_indices)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_dims, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_dims, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)
