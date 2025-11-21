import torch
from typing import Optional, Tuple

def rotate_sequence(P, Q):
    B, T, N, _ = P.shape

    # --- Step 1: centroids of first frame ---
    centroid_P0 = P[:, 0].mean(dim=1, keepdim=True)  # (B,1,3)
    centroid_Q = Q.mean(dim=1, keepdim=True)  # (B,1,3)

    P0_centered = P[:, 0] - centroid_P0  # (B,N,3)
    Q_centered = Q - centroid_Q  # (B,N,3)

    # --- Step 2: Procrustes on XY plane ---
    P_xy = P0_centered[:, :, :2]  # (B,N,2)
    Q_xy = Q_centered[:, :, :2]

    # import pdb; pdb.set_trace()
    H = torch.einsum("bni,bnj->bij", P_xy, Q_xy)  # (B,2,2)
    U, _, Vt = torch.linalg.svd(H)
    R2 = Vt @ U.transpose(-2, -1)  # (B,2,2)

    # Ensure det(R)=+1
    det = R2[...,0, 0]*R2[...,1, 1] - R2[...,0, 1]*R2[...,1, 0]
    # det = torch.linalg.det(R2)
    mask = det < 0
    if mask.any():
        Vt[mask, 1, :] *= -1
        R2 = Vt @ U.transpose(-2, -1)

    # --- Step 3: embed into 3D rotation matrix ---
    Rz = torch.eye(3, device=P.device, dtype=P.dtype).unsqueeze(0).repeat(B, 1, 1)
    Rz[:, :2, :2] = R2

    # --- Step 4: apply to whole sequence ---
    P_centered_seq = P - centroid_P0.unsqueeze(1)  # (B,T,N,3)
    P_rotated_seq = torch.einsum("bij,btnj->btni", Rz, P_centered_seq)

    # Compute translation so first frame matches
    t = centroid_Q.squeeze(1) - P_rotated_seq[:,0].mean(dim=1)
    # t = Q[:, 0] - P_rotated_seq[:, 0, 0]#.mean(dim=1)  # (B,3)
    P_aligned = P_rotated_seq + t[:, None, None, :]

    return P_aligned

def rotate_spine(
    pose: torch.Tensor,
    vector: Tuple = (4, 3),
    lock_to_x: bool = False,
) -> torch.Tensor:
    """
    Rotate poses so that the vector from root -> forward lies in the xz-plane
    (yaw-only correction, no pitch adjustment).

    Parameters
    ----------
    pose : (F, J, 3) torch.Tensor
        Sequence of 3D poses (F frames, J joints, 3 coordinates).
    anchor_indices : tuple of int, optional
        Indices (root, forward) used to define spine direction.
        Ignored if forward_vectors is given.
    forward_vectors : (F, 3) torch.Tensor, optional
        Precomputed forward vectors per frame.

    Returns
    -------
    pose_rot : (F, J, 3) torch.Tensor
        Rotated poses.
    """
    num_joints = pose.shape[1]
    if len(vector) == 2:
        yaw = -torch.arctan2(
            pose[:, vector[1], 1], pose[:, vector[1], 0]
        )  # Find angle to rotate to axis
    else:
        yaw = -torch.arctan2(vector[:, 1], vector[:, 0])
        # Find angle to rotate to axis

    if lock_to_x:
        print("Rotating spine to x axis ... ")
        if len(vector) == 2:
            pitch = torch.arctan2(pose[:, vector[1], 2], pose[:, vector[1], 0])
        else:
            pitch = torch.arctan2(vector[:, 2], vector[:, 0])
    else:
        print("Rotating spine to xz plane ... ")
        pitch = torch.zeros(yaw.shape, dtype=pose.dtype, device=pose.device)

    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    rot_mat = torch.zeros((3, 3, len(yaw)), dtype=yaw.dtype, device=yaw.device)
    rot_mat[0, 0] = cos_yaw
    rot_mat[0, 1] = -sin_yaw
    rot_mat[1, 0] = sin_yaw
    rot_mat[1, 1] = cos_yaw
    rot_mat[2, 2] = 1.0
    rot_mat = rot_mat.repeat_interleave(num_joints, dim=2)
    pose_rot = torch.einsum("jki,ik->ij", rot_mat, torch.reshape(pose, (-1, 3))).reshape(
        pose.shape
    )

    return pose_rot