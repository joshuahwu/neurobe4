import torch
from typing import Optional, Tuple
from scrubvae.data import quaternion as qtn
from scrubvae.data.dataset import inv_kin_torch, fwd_kin_cont6d_torch
from scrubvae.data.rotation_conversion import matrix_to_quaternion

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
    det = R2[..., 0, 0] * R2[..., 1, 1] - R2[..., 0, 1] * R2[..., 1, 0]
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
    t = centroid_Q.squeeze(1) - P_rotated_seq[:, 0].mean(dim=1)
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
    pose_rot = torch.einsum(
        "jki,ik->ij", rot_mat, torch.reshape(pose, (-1, 3))
    ).reshape(pose.shape)

    return pose_rot


def get_yaw_from_mat(
    mat: torch.Tensor,
):
    fwd_matrix = torch.matmul(mat[..., 0, :, :], mat[..., 1, :, :])
    yaw = torch.atan2(fwd_matrix[..., 1, 0], fwd_matrix[..., 0, 0])

    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)

    yaw_mat = torch.stack(
        [
            torch.stack([cy, -sy, zeros], dim=-1),
            torch.stack([sy, cy, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )

    return yaw_mat


def remove_yaw_from_cont6d(cont6d: torch.Tensor, root: torch.Tensor):
    window = cont6d.shape[-3]
    rot_mat = qtn.cont6d_to_matrix(cont6d, eps=1e-6)
    yaw_mat = get_yaw_from_mat(rot_mat[..., window // 2, :, :, :])
    yaw_mat = yaw_mat.unsqueeze_(-3)
    yaw_mat = yaw_mat.repeat_interleave(window, dim=-3)

    # Apply inverse yaw rotation to the rotation matrix
    inv_yaw_mat = torch.transpose(yaw_mat, -2, -1)
    rot_mat[..., 0, :, :] = torch.matmul(inv_yaw_mat, rot_mat[..., 0, :, :])
    # root_no_yaw = torch.matmul(inv_yaw_mat, root)
    root_no_yaw = torch.einsum("btij,btj->bti", inv_yaw_mat, root)

    # Convert back to 6D representation
    cont6d_no_yaw = rot_mat[..., :2].swapaxes(-1, -2).reshape(rot_mat.shape[:-2] + (6,))

    return cont6d_no_yaw, root_no_yaw


def process_batch_output(data_o: dict, kinematic_tree: dict, offsets: torch.Tensor):
    window = data_o["x6d"].shape[-3]
    data_o["root"][..., [0, 1]] -= data_o["root"][..., window//2: window//2 + 1, [0, 1]]
    data_o["x6d"], data_o["root"] = fwd_kin_root(data_o["x6d"], data_o["root"], kinematic_tree, offsets)
    # data_o["x6d"], data_o["root"] = remove_yaw_from_cont6d(data_o["x6d"], data_o["root"])
    return data_o

def fwd_kin_root(cont6d: torch.Tensor, root: torch.Tensor, kinematic_tree: dict, offsets: torch.Tensor):
    window = cont6d.shape[-3]
    n_keypts = cont6d.shape[-2]
    pose = fwd_kin_cont6d_torch(
        cont6d.reshape((-1, n_keypts, 6)),
        kinematic_tree,
        offsets,
        torch.zeros((cont6d.shape[0] * window, 3), device=cont6d.device),
        do_root_R=True,
        eps=1e-8,
    ).reshape(cont6d.shape[:-1] + (3,))

    try:
        assert not torch.any(torch.norm((pose[:, :, 1] - pose[:, :, 0]), dim=-1)==0)
    except:
        import pdb; pdb.set_trace()
    

    local_qtn = inv_kin_torch(
            pose.reshape((-1,n_keypts, 3)),
            kinematic_tree,
            offsets,
            forward_indices=[1, 0],
        ).reshape(pose.shape[:-1] + (-1,))

    fwd_vec = pose[:, window // 2, 1, :] - pose[:, window // 2, 0, :]
    fwd_vec = torch.nan_to_num(fwd_vec / torch.norm(fwd_vec, dim=-1, keepdim=True), nan=0.0)
    yaw = -torch.arctan2(fwd_vec[:, 1], fwd_vec[:, 0])
    yaw2D = torch.cat(
        [torch.sin(yaw / 2)[:, None], torch.cos(yaw / 2)[:, None]], axis=-1
    )

    fwd_qtn = torch.zeros((len(yaw), 4), dtype=cont6d.dtype, device=cont6d.device)
    fwd_qtn[:, [-1, 0]] = yaw2D
    fwd_qtn = torch.repeat_interleave(fwd_qtn[:, None], window, dim=1)
    local_qtn[..., 0, :] = qtn.qmul(fwd_qtn, local_qtn[..., 0, :])

    root = qtn.qrot(fwd_qtn, root)
    cont6d = qtn.quaternion_to_cont6d(local_qtn)
    return cont6d, root

# def fwd_kin_root(cont6d: torch.Tensor, root: torch.Tensor, kinematic_tree: dict, offsets: torch.Tensor):
#     window = cont6d.shape[-3]
#     root_matR = qtn.cont6d_to_matrix(cont6d[..., 0, :]).reshape(-1, 3, 3)
    
#     matR = torch.matmul(root_matR, qtn.cont6d_to_matrix(cont6d[..., 1, :]).reshape(-1, 3, 3))
#     offsets = torch.tensor([1, 0, 0], device=cont6d.device, dtype=cont6d.dtype).expand(
#         matR.shape[0], -1
#     )

#     fwd_vec = torch.matmul(matR, offsets[..., None]).squeeze(-1)
#     fwd_vec = fwd_vec / torch.norm(fwd_vec, dim=-1, keepdim=True)
#     target = torch.tensor([1, 0, 0], device=cont6d.device, dtype=cont6d.dtype).repeat(
#         len(fwd_vec), 1
#     )
#     root_quat = qtn.qbetween(-fwd_vec, target)

#     cont6d[..., 1, :] = qtn.quaternion_to_cont6d(
#         qtn.qmul(qtn.qinv(root_quat), qtn.qbetween(target, fwd_vec))
#     ).reshape(cont6d.shape[:-2] + (6,))

#     fwd_vec = fwd_vec.reshape(-1, window, 3)
#     yaw = -torch.arctan2(fwd_vec[:, window // 2, 1], fwd_vec[:, window // 2, 0])
#     yaw2D = torch.cat(
#         [torch.sin(yaw / 2)[:, None], torch.cos(yaw / 2)[:, None]], axis=-1
#     )
#     fwd_qtn = torch.zeros((len(yaw), 4), dtype=cont6d.dtype, device=cont6d.device)
#     fwd_qtn[:, [-1, 0]] = yaw2D
#     fwd_qtn = torch.repeat_interleave(fwd_qtn, window, dim=0)
#     cont6d[..., 0, :] = qtn.quaternion_to_cont6d(
#         qtn.qmul(fwd_qtn, root_quat)
#     ).reshape(cont6d.shape[:-2] + (6,))

#     spine_L_matR = qtn.cont6d_to_matrix(cont6d[..., 5, :]).reshape(-1, 3, 3)
#     spine_L_matR = torch.matmul(root_matR, spine_L_matR)
#     spine_L_matR = torch.matmul(torch.linalg.inv(qtn.cont6d_to_matrix(cont6d[..., 0, :])).reshape(-1, 3, 3), spine_L_matR)
#     cont6d[..., 5, :] = spine_L_matR[..., :2].swapaxes(-1, -2).reshape((-1, window, 6,))

#     root = qtn.qrot(fwd_qtn, root.reshape(-1, 3)).reshape(root.shape)

#     return cont6d, root

