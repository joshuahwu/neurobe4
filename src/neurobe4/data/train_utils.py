import torch
import torch.nn.functional as F
from typing import Dict, Optional
from scrubvae.data import quaternion as qtn
from scrubvae.data.dataset import inv_kin_torch, fwd_kin_cont6d_torch


def _augment_enabled(augment_dict: Optional[Dict], key: str) -> bool:
    """True if augment_dict[key] is an active config -- not missing, not None,
    and not a dict with any None values (e.g. `noise: {magnitude: null}`).
    Copied from befound.data.train_utils."""
    if not augment_dict:
        return False
    value = augment_dict.get(key)
    if value is None:
        return False
    if isinstance(value, dict) and any(v is None for v in value.values()):
        return False
    return True


def get_midfwd_rotation(
    pose: torch.Tensor, kinematic_tree: list, skeleton_offsets: torch.Tensor
):
    """
    Adapted from befound.data.train_utils.get_midfwd_rotation. Each window is
    translated so its middle frame's root is at the xy origin and rotated so
    its middle frame's root -> forward (keypoint 0 -> 1) direction faces x+.

    Parameters
    ----------
    pose : (B, W, N_K, 3) torch.Tensor
    kinematic_tree : list
    skeleton_offsets : (N_K, 3) torch.Tensor

    Returns
    -------
    x6d : (B, W, N_K, 6) torch.Tensor
        Local joint rotations in continuous 6D form.
    root : (B, W, 3) torch.Tensor
        Root position. Only x/y are centered (z keeps its height), as in
        neurobe4's preprocess_time_paired, since models normalize root with
        arena_size.
    """
    B, W, N_K, D = pose.shape
    forward = pose[:, W // 2, 1, :] - pose[:, W // 2, 0, :]
    yaw = -torch.atan2(forward[:, 1], forward[:, 0])[:, None]

    root = pose[..., 0, :].clone()
    root[..., :2] -= root[:, W // 2 : W // 2 + 1, :2].clone()

    local_qtn = inv_kin_torch(
        pose.reshape((-1, N_K, D)),
        kinematic_tree,
        skeleton_offsets,
        forward_indices=[1, 0],
    ).reshape(B, W, N_K, 4)

    # Yaw rotation about z as a quaternion (w, x, y, z) = (cos(yaw/2), 0, 0, sin(yaw/2))
    fwd_qtn = torch.zeros((B, 4), dtype=pose.dtype, device=pose.device)
    fwd_qtn[:, [-1, 0]] = torch.cat([torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=-1)
    fwd_qtn = fwd_qtn.unsqueeze(1).repeat(1, W, 1)
    local_qtn[..., 0, :] = qtn.qmul(fwd_qtn, local_qtn[..., 0, :])

    root = qtn.qrot(fwd_qtn, root)
    x6d = qtn.quaternion_to_cont6d(local_qtn)

    return x6d, root


def relative_pose(target: torch.Tensor, partner: torch.Tensor) -> torch.Tensor:
    """
    The partner's position and heading as seen by the target animal: in the
    target's egocentric frame (root at the origin, root -> forward keypoint
    0 -> 1 along x+, as in get_midfwd_rotation).

    Parameters
    ----------
    target, partner : (..., N_K >= 2, 3) torch.Tensor, raw poses at the same frames

    Returns
    -------
    (..., 4) torch.Tensor: dx, dy of the partner's root, and the cos and sin
    of its heading relative to the target's (continuous, unlike the angle)
    """
    # Unit heading vectors (root -> forward keypoint)
    u = F.normalize(target[..., 1, :2] - target[..., 0, :2], dim=-1)
    v = F.normalize(partner[..., 1, :2] - partner[..., 0, :2], dim=-1)

    d = partner[..., 0, :2] - target[..., 0, :2]
    # Partner offset rotated into the target's frame (u along x+)
    dx = u[..., 0] * d[..., 0] + u[..., 1] * d[..., 1]
    dy = -u[..., 1] * d[..., 0] + u[..., 0] * d[..., 1]
    # cos / sin of the angle from u to v
    cos = u[..., 0] * v[..., 0] + u[..., 1] * v[..., 1]
    sin = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    return torch.stack([dx, dy, cos, sin], dim=-1)


def prepare_batch(
    data: Dict,
    dataset,
    augment_dict: Optional[Dict] = None,
    device: str = "cuda",
) -> Dict:
    """
    Convert a raw-pose batch from neurobe4.data.dataset.RodentDiskDataset,
    RodentDiskDatasetTPaired or SocialDiskDatasetTPaired into the model/loss
    inputs. Adapted from befound.data.train_utils.prepare_batch for a single
    dataset.

    Each window is independently centered and rotated (midfwd) before
    inverse kinematics -- the partner's windows too, in their own frames;
    where the partner is relative to the target is given by rel_pose.

    Parameters
    ----------
    data : dict
        Batch with "pose" of shape (B, W, N_K, 3) (RodentDiskDataset) or
        (B, T, W, N_K, 3) (RodentDiskDatasetTPaired), plus "ids" and
        optionally "tau". Social batches (SocialDiskDatasetTPaired) also have
        "pose_partner" (B, T, W, N_K, 3), "target", and "vae_z_partner" if
        latents are attached.
    dataset : RodentDiskDataset
        Supplies kinematic_tree, skeleton_offsets and get_default_offsets.
    augment_dict : dict, optional
        Augmentations (pass only during training). Supported:
            noise: {magnitude: float} -- Gaussian noise with std `magnitude`
                (in raw pose units) added to the model *input* pose before
                inverse kinematics; target_pose is computed from the clean
                pose. Not supported for time-paired batches (raises
                ValueError).

    Returns
    -------
    dict with keys:
        x6d : (B, [T,] W, N_K, 6)
        root : (B, [T,] W, 3)
        offsets : (N_K, 3) default offsets
        target_pose : (B, W, N_K, 3) root-centered pose from clean x6d with
            default offsets -- of the single window, or of window 1 (the
            first target window) if time-paired, matching MouseDatasetTPaired
        ids, tau, vae_z, target, vae_z_partner (if present) : passed through
        x6d_partner : (B, T, W, N_K, 6), social batches only
        root_partner : (B, T, W, 3), social batches only
        rel_pose : (B, 4), social batches only -- the partner's dx, dy and
            cos, sin of its relative heading in the target's egocentric frame
            (relative_pose)
            at the one frame shared by the past window 0 and the target
            window 1 (requires n_skip = window - 1)
    If the batch has no "pose" (latents only), only offsets are added.
    """
    offsets = dataset.get_default_offsets(device)
    assert offsets is not None, "prepare_batch requires use_default_seg_len=True"
    kinematic_tree = dataset.kinematic_tree
    skeleton_offsets = dataset.skeleton_offsets.to(device)

    data = {k: v.to(device) for k, v in data.items()}
    if "pose" not in data:
        # Latent-only batch (dataset.load_pose=False); "vae_z" passes through
        data["offsets"] = offsets
        return data
    pose = data.pop("pose")
    is_paired = pose.dim() == 5
    if is_paired and _augment_enabled(augment_dict, "noise"):
        raise ValueError("noise augmentation is not supported for time-paired data")
    lead_shape = pose.shape[:-3]  # (B,) or (B, T)
    W, N_K = pose.shape[-3], pose.shape[-2]
    pose_partner = data.pop("pose_partner", None)  # social batches

    # Both animals' windows through one inverse-kinematics call (each window
    # still in its own midfwd frame), then split back
    windows = pose.reshape((-1, W, N_K, 3))
    if pose_partner is not None:
        windows = torch.cat([windows, pose_partner.reshape((-1, W, N_K, 3))])
    x6d, root = get_midfwd_rotation(windows, kinematic_tree, skeleton_offsets)
    if pose_partner is not None:
        x6d, x6d_partner = x6d.chunk(2)
        root, root_partner = root.chunk(2)
    x6d = x6d.reshape(lead_shape + (W, N_K, 6))
    root = root.reshape(lead_shape + (W, 3))

    # Target Pose (clean)
    target_x6d = x6d[:, 1] if is_paired else x6d
    B = target_x6d.shape[0]
    data["target_pose"] = fwd_kin_cont6d_torch(
        target_x6d.reshape((-1, N_K, 6)),
        kinematic_tree,
        offsets,
        root_pos=torch.zeros((B * W, 3), dtype=x6d.dtype, device=device),
        do_root_R=True,
        eps=1e-8,
    ).reshape((B, W, N_K, 3))

    if _augment_enabled(augment_dict, "noise"):
        # Re-derive the input from a noised copy of the raw pose
        noisy_pose = pose + augment_dict["noise"]["magnitude"] * torch.randn_like(pose)
        x6d, root = get_midfwd_rotation(noisy_pose, kinematic_tree, skeleton_offsets)

    data["x6d"] = x6d
    data["root"] = root
    data["offsets"] = offsets

    if pose_partner is not None:
        if dataset.n_skip != W - 1:
            raise ValueError(
                f"rel_pose uses the frame shared by windows 0 and 1, which needs "
                f"n_skip = window - 1 ({W - 1}), got n_skip = {dataset.n_skip}"
            )
        # Last frame of window 0 = first frame of window 1
        data["rel_pose"] = relative_pose(pose[:, 0, -1], pose_partner[:, 0, -1])
        data["x6d_partner"] = x6d_partner.reshape(lead_shape + (W, N_K, 6))
        data["root_partner"] = root_partner.reshape(lead_shape + (W, 3))

    return data
