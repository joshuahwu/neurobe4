from neurobe4.data.constants import (
    OFFSETS_3D_PAIRR24M,
    OFFSETS_3D_MOUSE,
    OFFSETS_3D,
)
from neuroposelib import read
import numpy as np
import scrubvae.data.quaternion as qtn
from typing import Optional, Type, Union, List, Dict
from torch.utils.data import Dataset
import torch
import h5py
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import trange
from scrubvae.data.dataset import (
    get_speed_outliers,
    get_speed_parts,
    get_frame_yaw,
    get_angle2D,
    inv_kin,
    get_segment_len,
    get_window_indices,
    fwd_kin_cont6d_torch,
)
import pandas as pd
import re

# Datasets of animal pairs: time-aligned pose_a1.h5 / pose_a2.h5 (social
# sessions) plus pose_single.h5 (single-animal sessions)
SOCIAL_DATASETS = ("pairr24m", "sdannce")

# Suffix of the filtered window-center index files (use_filtered_inds) and of
# the caches built from them (cache_name). Earlier "_filtered" index files
# (and caches named "_filtered") are a different window set.
FILTERED_SUFFIX = "_filtered_offsets"


def preprocess_time_paired(
    data_path: str,
    skeleton_config: dict,
    dataset: str,
    window: int,
    n_skip: int = 25,
    train_val_test: str = "train",
    data_keys: List[str] = ["x6d", "root", "offsets"],
    speed_threshold: Optional[float] = 2.25,
    direction_process: str = "midfwd",
    n_time_steps: int = 1,
    use_default_offsets: bool = False,
    demo: bool = False,
):
    """Prepare and save all data preprocessing for SC-VAE model training, validation, and testing

    Parameters
    ----------
    data_path : str
        Path to folder with datasets
    skeleton_config : dict
        Configuration file denoting the structure of the skeleton
    dataset : str
        Which dataset to prepare (i.e., "4_mice", "parkinsons")
    data_keys : List[str], optional
        Keys of data to save, by default ["x6d", "root", "offsets"]
    speed_threshold : Optional[float], optional
        Action segments with greater average speed will be filtered out, by default 2.25
    direction_process : str, optional
        Preprocess pose sequences such that the animals pass through the origin at the middle frame from
        any direction ("x360") or only in the x+ direction ("midfwd"), by default "midfwd"
    use_default_offsets : bool, optional
        Whether to use default segment lengths for offsets, by default False

    Returns
    -------
    data
        Dictionary with key-value pairs associated with `data_keys`
    """
    dataset_ids = {
        "parkinsons": 72,
        "4_mice": 4,
        "parkinsons_healthy": 72,
        "pairr24m": 40,
    }
    dataset_strides = {
        "parkinsons": 10,
        "4_mice": 1,
        "parkinsons_healthy": 10,
        "pairr24m": 10,
    }
    n_ids = dataset_ids[dataset]
    n_stride = dataset_strides[dataset]  # 10 if "parkinsons" in dataset else 1
    print("Calculating dataset: {}".format(dataset))
    if dataset == "parkinsons_healthy":
        dataset_name = "parkinsons"
    else:
        dataset_name = dataset
    # Already windowed and split by dataset
    pose = np.load("{}{}/{}/pose.npy".format(data_path, dataset_name, train_val_test))
    ids = np.repeat(np.arange(n_ids), len(pose) // n_ids)

    if dataset == "parkinsons_healthy":
        pose = pose[ids < 36, ...]
        ids = ids[ids < 36]
        n_ids = 36

    n_ids = 4 if demo else n_ids

    if demo:
        pose = pose[ids < 4]
        ids = ids[ids < 4]

    n_min_sections = 30 if train_val_test == "train" else 15  # we did 50-25-25 split
    shape = (
        n_ids,
        n_min_sections,
        -1,
    )

    split_inds = np.arange(len(pose)).reshape(shape)
    assert n_skip // n_stride < split_inds.shape[2]
    t_paired_inds = []
    len_sec = split_inds.shape[-1]
    for t in range(n_time_steps + 1):
        start_i = t * n_skip // n_stride
        end_i = len_sec - (n_time_steps - t) * n_skip // n_stride
        t_paired_inds += [split_inds[:, :, start_i:end_i].flatten()]
        # past_inds = split_inds[:, :, :-n_skip//n_stride].flatten()
        # future_inds = split_inds[:, :, n_skip//n_stride:].flatten()
    t_paired_inds = np.vstack(t_paired_inds).T

    # Filter out bad tracking using speed threshold
    if speed_threshold is not None:
        outlier_frames = get_speed_outliers(pose, speed_threshold)
        # convert to the paired indices
        outlier_inds = np.unique(np.where(np.isin(t_paired_inds, outlier_frames))[0])
        t_paired_inds = np.delete(t_paired_inds, outlier_inds, 0)

    data_len = len(pose)
    data = {}  # "raw_pose": pose}
    # Get yaw of mid-spine -> fwd-spine segment for central frame in all windows
    yaw = get_frame_yaw(pose[:, window // 2, ...], 0, 1)[..., None]

    if ("root" or "x6d") in data_keys:
        root = pose[..., 0, :].copy()  # (n_samples, window, 3)
        if direction_process in ["midfwd", "x360"]:
            # Centering root of middle frame
            root_center = np.zeros(root.shape)  # (n_samples, window, 3)
            root_center[..., [0, 1]] = root[:, window // 2, [0, 1]][
                :, None, :
            ]  # (n_samples, 1, 2)

            root -= root_center

    # Applying inverse kinematics to get local quaternions
    # That representation is then converted to a continuous 6D rotation
    if "x6d" in data_keys:
        print("Applying inverse kinematics ...")
        local_qtn = inv_kin(
            pose.reshape((-1,) + pose.shape[-2:]),
            skeleton_config["KINEMATIC_TREE"],
            np.array(skeleton_config["OFFSET"]),
            forward_indices=[1, 0],
        ).reshape(pose.shape[:-1] + (-1,))

        if direction_process == "midfwd":
            ## Center frame of a window is translated to center and rotated to x+
            fwd_qtn = np.zeros((len(yaw), 4))
            fwd_qtn[:, [-1, 0]] = get_angle2D(yaw / 2)
            fwd_qtn = np.repeat(fwd_qtn[:, None, :], window, axis=1)
            local_qtn[..., 0, :] = qtn.qmul_np(fwd_qtn, local_qtn[..., 0, :])

            if "root" in data_keys:
                root = qtn.qrot_np(fwd_qtn, root)

        data["x6d"] = qtn.quaternion_to_cont6d_np(local_qtn)

    # Get offsets scaled by segment lengths
    if "offsets" in data_keys:
        if use_default_offsets:
            data["offsets"] = OFFSETS_3D_MOUSE[:, None] * np.array(
                skeleton_config["OFFSET"], dtype=np.float32
            )
        else:
            data["offsets"] = get_segment_len(
                pose.reshape((-1,) + pose.shape[-2:]),
                skeleton_config["KINEMATIC_TREE"],
                np.array(skeleton_config["OFFSET"], dtype=pose.dtype),
            ).reshape(pose.shape)
            # import pdb; pdb.set_trace()

    # Get root positions
    if "root" in data_keys:
        data["root"] = root
        frame_dim_inds = tuple(range(len(root.shape) - 1))
        print("Root Maxes: {}".format(root.max(axis=frame_dim_inds)))
        print("Root Mins: {}".format(root.min(axis=frame_dim_inds)))

    data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}

    # Get animal IDs
    if "ids" in data_keys:
        data["ids"] = torch.tensor(ids, dtype=torch.int16)

    if "target_pose" in data_keys:
        # Target pose root does not move
        reshaped_x6d = data["x6d"].reshape((-1,) + data["x6d"].shape[-2:])
        if use_default_offsets:
            offsets = data["offsets"]
        else:
            offsets = data["offsets"].reshape(reshaped_x6d.shape[:2] + (-1,))
        data["target_pose"] = fwd_kin_cont6d_torch(
            reshaped_x6d,
            skeleton_config["KINEMATIC_TREE"],
            offsets,
            root_pos=torch.zeros(reshaped_x6d.shape[0], 3),
            do_root_R=True,
            eps=1e-8,
        ).reshape(data["x6d"].shape[:-1] + (3,))

    for k, v in data.items():
        try:
            assert len(v) == data_len
        except:
            assert (len(v) == pose.shape[-2]) and (k == "offsets")

    t_paired_inds = torch.tensor(t_paired_inds, dtype=torch.long)
    return data, t_paired_inds


class MouseDatasetTPaired(Dataset):
    """
    Dataset class for Mouse dataset
    """

    def __init__(
        self,
        data,
        paired_inds,
        arena_size=None,
        kinematic_tree=None,
        n_keypts=None,
        label="train",
        discrete_classes=None,
        norm_params=None,
    ):
        self.data_keys = list(data.keys())
        self.data = data
        self.n_keypts = n_keypts
        self.discrete_classes = discrete_classes
        self.norm_params = norm_params
        self.paired_inds = paired_inds
        # self.n_skip = paired_inds[0, 1] - paired_inds[0, 0]

        if arena_size is not None:
            self.arena_size = torch.tensor(arena_size)
        else:
            self.arena_size = None

        self.kinematic_tree = kinematic_tree
        self.label = label
        self.standard_offsets = False
        if "offsets" in self.data_keys:
            if self.data["offsets"].ndim == 2:
                self.standard_offsets = True

    def __len__(self):
        return len(self.paired_inds)

    def __getitem__(self, idx):
        # Query items which have already been windowed
        query = {}
        for k, v in self.data.items():
            if k in ["target_pose", "pd_label", "ids"]:
                query[k] = v[self.paired_inds[idx, 1]]
            elif k in ["offsets"]:
                if self.standard_offsets:
                    continue
                else:
                    query[k] = v[self.paired_inds[idx, :]]
            elif k in ["vae_z"]:
                query[k] = v[idx]
            else:
                query[k] = v[self.paired_inds[idx, :]]

        return query


def preprocess_pairr24m_data(
    data_path: str,
    skeleton_config: dict,
    dataset: str,
    window: int,
    train_val_test: str = "train",
    stride: int = 1,
    data_keys: List[str] = ["x6d", "root", "offsets"],
    direction_process: str = "midfwd",
    use_default_offsets: bool = False,
    get_social_paired: bool = False,
    demo: bool = False,
):
    """Prepare and save all data preprocessing for SC-VAE model training, validation, and testing

    Parameters
    ----------
    data_path : str
        Path to folder with datasets
    skeleton_config : dict
        Configuration file denoting the structure of the skeleton
    dataset : str
        Which dataset to prepare (i.e., "4_mice", "parkinsons")
    data_keys : List[str], optional
        Keys of data to save, by default ["x6d", "root", "offsets"]
    direction_process : str, optional
        Preprocess pose sequences such that the animals pass through the origin at the middle frame from
        any direction ("x360") or only in the x+ direction ("midfwd"), by default "midfwd"
    use_default_offsets : bool, optional
        Whether to use default segment lengths for offsets, by default False

    Returns
    -------
    data
        Dictionary with key-value pairs associated with `data_keys`
    """

    meta = pd.read_csv(data_path + "metadata.csv")

    labels_list = [
        "goodFrame_an1",
        "goodFrame_an2",
        "interactionCat",
        "behaviorFine_an1",
        "behaviorFine_an2",
        "behaviorCoarse_an1",
        "behaviorCoarse_an2",
    ]
    pose_a1, ids = read.pose_h5(data_path + "pose_a1.h5")
    pose_a2, _ = read.pose_h5(data_path + "pose_a2.h5")
    split_inds = np.load(data_path + "{}_inds.npy".format(train_val_test))
    pose_a1 = pose_a1.astype(np.float32)
    pose_a2 = pose_a2.astype(np.float32)
    split_inds = split_inds.astype(np.int32)
    ids = ids.astype(np.int16)

    curr_id = 0
    split_ids = []
    for i in np.unique(ids):
        split_inds_i = split_inds[ids[split_inds] == i]
        n_consecutive = np.where(
            np.diff(split_inds_i, prepend=split_inds_i[0] - 1) != 1
        )[0]
        n_consecutive = np.diff(n_consecutive, prepend=0, append=len(split_inds_i))
        split_ids += [np.repeat(np.arange(len(n_consecutive)) + curr_id, n_consecutive)]
        curr_id += len(n_consecutive)

    split_ids = np.concatenate(split_ids)
    # labels_dict = {k: [] for k in labels_list}
    # for i, f_path in enumerate(meta["FolderPath"]):
    #     markers_csv = pd.read_csv(data_path + f_path + "/markerDataset.csv")
    #     # for label in labels_list:
    #     #     labels_dict[label].append(markers[label].to_numpy())
    # labels_dict = {
    #     k: torch.from_numpy(np.concatenate(v, axis=0)) for k, v in labels_dict.items()
    # }

    # for label in ["goodFrame", "behaviorFine", "behaviorCoarse"]:
    #     snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", label).lower()
    #     if snake_case in data_keys:
    #         labels_dict[re.sub(r"(?<!^)(?=[A-Z])", "_", label).lower()] = torch.vstack(
    #             (labels_dict[label + "_an1"], labels_dict[label + "_an2"])
    #         )
    #         del labels_dict[label + "_an1"]
    #         del labels_dict[label + "_an2"]

    window_inds = get_window_indices(split_ids, stride, window)

    if demo:
        window_inds = np.concatenate(
            [
                window_inds[:20000, ...],
                window_inds[len(window_inds) // 2 : len(window_inds) // 2 + 20000],
            ],
            axis=0,
        )
    ids = ids[split_inds[window_inds]][:, window // 2]
    pose_a1 = pose_a1[split_inds[window_inds]]
    pose_a2 = pose_a2[split_inds[window_inds]]

    # pose = pose[window_inds, ...].astype(np.float32)
    # ids = ids[window_inds][:, window // 2]

    pose = np.concatenate([pose_a1, pose_a2], axis=0)
    ids = np.concatenate([ids, ids + ids.max() + 1])

    print("Calculating dataset: {}".format(dataset))

    # Filter out bad tracking given nan lables
    nan_frames = np.unique(np.where(np.isnan(pose))[0])
    pose = np.delete(pose, nan_frames, axis=0)
    ids = np.delete(ids, nan_frames, axis=0)

    data_len = len(pose)
    data = {}  # "raw_pose": pose}

    # Get yaw of mid-spine -> fwd-spine segment for central frame in all windows
    yaw = get_frame_yaw(pose[:, window // 2, ...], 0, 1)[..., None]

    if ("root" or "x6d") in data_keys:
        root = pose[..., 0, :].copy()  # (n_samples, window, 3)
        if direction_process in ["midfwd", "x360"]:
            # Centering root of middle frame
            root_center = np.zeros(root.shape)  # (n_samples, window, 3)
            root_center[..., [0, 1]] = root[:, window // 2, [0, 1]][
                :, None, :
            ]  # (n_samples, 1, 2)

            root -= root_center

    # Applying inverse kinematics to get local quaternions
    # That representation is then converted to a continuous 6D rotation
    if "x6d" in data_keys:
        print("Applying inverse kinematics ...")
        local_qtn = inv_kin(
            pose.reshape((-1,) + pose.shape[-2:]),
            skeleton_config["KINEMATIC_TREE"],
            np.array(skeleton_config["OFFSET"]),
            forward_indices=[1, 0],
        ).reshape(pose.shape[:-1] + (-1,))

        if direction_process == "midfwd":
            ## Center frame of a window is translated to center and rotated to x+
            fwd_qtn = np.zeros((len(yaw), 4))
            fwd_qtn[:, [-1, 0]] = get_angle2D(yaw / 2)
            fwd_qtn = np.repeat(fwd_qtn[:, None, :], window, axis=1)
            local_qtn[..., 0, :] = qtn.qmul_np(fwd_qtn, local_qtn[..., 0, :])

            if "root" in data_keys:
                root = qtn.qrot_np(fwd_qtn, root)

        data["x6d"] = qtn.quaternion_to_cont6d_np(local_qtn)

    # Get offsets scaled by segment lengths
    if "offsets" in data_keys:
        if use_default_offsets:
            data["offsets"] = OFFSETS_3D_PAIRR24M[:, None] * np.array(
                skeleton_config["OFFSET"], dtype=np.float32
            )
        else:
            data["offsets"] = get_segment_len(
                pose.reshape((-1,) + pose.shape[-2:]),
                skeleton_config["KINEMATIC_TREE"],
                np.array(skeleton_config["OFFSET"]),
            ).reshape(pose.shape)

    # Get root positions
    if "root" in data_keys:
        data["root"] = root
        frame_dim_inds = tuple(range(len(root.shape) - 1))
        print("Root Maxes: {}".format(root.max(axis=frame_dim_inds)))
        print("Root Mins: {}".format(root.min(axis=frame_dim_inds)))

    data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}

    if "target_pose" in data_keys:
        # Target pose root does not move
        reshaped_x6d = data["x6d"].reshape((-1,) + data["x6d"].shape[-2:])
        if use_default_offsets:
            offsets = data["offsets"]
        else:
            offsets = data["offsets"].reshape(reshaped_x6d.shape[:2] + (-1,))
        data["target_pose"] = fwd_kin_cont6d_torch(
            reshaped_x6d,
            skeleton_config["KINEMATIC_TREE"],
            offsets,
            root_pos=torch.zeros(reshaped_x6d.shape[0], 3),
            do_root_R=True,
            eps=1e-8,
        ).reshape(data["x6d"].shape[:-1] + (3,))

    # if get_paired:
    #     data = {
    #         k: v.reshape((2, -1, *v.shape[1:])) if k != "offsets" else v
    #         for k, v in data.items()
    #     }
    # data.update(labels_dict)

    # Get animal IDs
    if "ids" in data_keys:
        data["ids"] = torch.tensor(ids, dtype=torch.int16)

    for k, v in data.items():
        try:
            assert len(v) == data_len
        except:
            assert (len(v) == pose.shape[-2]) and (k == "offsets")

    return data


def preprocess_time_paired_pairr24m(
    data_path: str,
    skeleton_config: dict,
    dataset: str,
    window: int,
    train_val_test: str = "train",
    stride: int = 1,
    data_keys: List[str] = ["x6d", "root", "offsets"],
    direction_process: str = "midfwd",
    use_default_offsets: bool = False,
    n_skip: int = 50,
    n_time_steps: int = 1,
    demo: bool = False,
):
    """Prepare and save all data preprocessing for SC-VAE model training, validation, and testing

    Parameters
    ----------
    data_path : str
        Path to folder with datasets
    skeleton_config : dict
        Configuration file denoting the structure of the skeleton
    dataset : str
        Which dataset to prepare (i.e., "4_mice", "parkinsons")
    data_keys : List[str], optional
        Keys of data to save, by default ["x6d", "root", "offsets"]
    direction_process : str, optional
        Preprocess pose sequences such that the animals pass through the origin at the middle frame from
        any direction ("x360") or only in the x+ direction ("midfwd"), by default "midfwd"
    use_default_offsets : bool, optional
        Whether to use default segment lengths for offsets, by default False

    Returns
    -------
    data
        Dictionary with key-value pairs associated with `data_keys`
    """

    meta = pd.read_csv(data_path + "metadata.csv")

    labels_list = [
        "goodFrame_an1",
        "goodFrame_an2",
        "interactionCat",
        "behaviorFine_an1",
        "behaviorFine_an2",
        "behaviorCoarse_an1",
        "behaviorCoarse_an2",
    ]
    pose_a1, ids = read.pose_h5(data_path + "pose_a1.h5")
    pose_a2, _ = read.pose_h5(data_path + "pose_a2.h5")
    split_inds = np.load(data_path + "{}_inds.npy".format(train_val_test))
    pose_a1 = pose_a1.astype(np.float32)
    pose_a2 = pose_a2.astype(np.float32)
    split_inds = split_inds.astype(np.int32)
    ids = ids.astype(np.int16)

    curr_id = 0
    split_ids = []
    for i in np.unique(ids):
        split_inds_i = split_inds[ids[split_inds] == i]
        n_consecutive = np.where(
            np.diff(split_inds_i, prepend=split_inds_i[0] - 1) != 1
        )[0]
        n_consecutive = np.diff(n_consecutive, prepend=0, append=len(split_inds_i))
        split_ids += [np.repeat(np.arange(len(n_consecutive)) + curr_id, n_consecutive)]
        curr_id += len(n_consecutive)

    split_ids = np.concatenate(split_ids)
    # labels_dict = {k: [] for k in labels_list}
    labels_dict = {k: [] for k in data_keys if k in labels_list}
    for i, f_path in enumerate(meta["FolderPath"]):
        markers_csv = pd.read_csv(data_path + f_path + "/markerDataset.csv")
        for data_key in data_keys:
            if data_key in labels_list:
                labels_dict[data_key].append(markers_csv[data_key].to_numpy())
    labels_dict = {
        k: torch.from_numpy(np.concatenate(v, axis=0)) for k, v in labels_dict.items()
    }

    # for label in ["goodFrame", "behaviorFine", "behaviorCoarse"]:
    #     snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", label).lower()
    #     if snake_case in data_keys:
    #         labels_dict[re.sub(r"(?<!^)(?=[A-Z])", "_", label).lower()] = torch.vstack(
    #             (labels_dict[label + "_an1"], labels_dict[label + "_an2"])
    #         )
    #         del labels_dict[label + "_an1"]
    #         del labels_dict[label + "_an2"]

    window_inds = get_window_indices(split_ids, stride, window)
    if demo:
        window_inds = np.concatenate(
            [
                window_inds[:20000, ...],
                window_inds[len(window_inds) // 2 : len(window_inds) // 2 + 20000],
            ],
            axis=0,
        )
    ids = ids[split_inds[window_inds]][:, window // 2]
    pose_a1 = pose_a1[split_inds[window_inds]]
    pose_a2 = pose_a2[split_inds[window_inds]]
    labels_dict = {
        k: v[split_inds[window_inds]][:, window // 2] for k, v in labels_dict.items()
    }

    pose = np.concatenate([pose_a1, pose_a2], axis=0)
    ids = np.concatenate([ids, ids + ids.max() + 1])
    split_ids_expanded = np.concatenate(
        [
            split_ids[window_inds][:, window // 2],
            split_ids[window_inds][:, window // 2] + split_ids.max() + 1,
        ]
    )

    split_inds = np.arange(len(split_ids_expanded))
    t_paired_inds = []
    for i in np.unique(split_ids_expanded):
        split_inds_i = split_inds[split_ids_expanded == i]
        len_sec = len(split_inds_i)
        t_paired_inds_i = []
        for t in range(n_time_steps + 1):
            try:
                start_i = t * n_skip // stride
                end_i = len_sec - (n_time_steps - t) * n_skip // stride
                t_paired_inds_i += [split_inds_i[start_i:end_i].flatten()]
            except:
                import pdb; pdb.set_trace()

        t_paired_inds += [np.vstack(t_paired_inds_i).T]

    t_paired_inds = np.vstack(t_paired_inds)

    print("Calculating dataset: {}".format(dataset))
    # Filter out bad tracking given nan lables
    nan_frames = np.unique(np.where(np.isnan(pose))[0])
    t_paired_inds = np.delete(
        t_paired_inds,
        np.unique(np.where(np.isin(t_paired_inds, nan_frames))[0]),
        axis=0,
    )
    pose = np.nan_to_num(pose)
    # pose = np.delete(pose, nan_frames, axis=0)
    # ids = np.delete(ids, nan_frames, axis=0)

    data_len = len(pose)
    data = {}  # "raw_pose": pose}

    # Get yaw of mid-spine -> fwd-spine segment for central frame in all windows
    yaw = get_frame_yaw(pose[:, window // 2, ...], 0, 1)[..., None]

    if ("root" or "x6d") in data_keys:
        root = pose[..., 0, :].copy()  # (n_samples, window, 3)
        if direction_process in ["midfwd", "x360"]:
            # Centering root of middle frame
            root_center = np.zeros(root.shape)  # (n_samples, window, 3)
            root_center[..., [0, 1]] = root[:, window // 2, [0, 1]][
                :, None, :
            ]  # (n_samples, 1, 2)

            root -= root_center

    # Applying inverse kinematics to get local quaternions
    # That representation is then converted to a continuous 6D rotation
    if "x6d" in data_keys:
        print("Applying inverse kinematics ...")
        local_qtn = inv_kin(
            pose.reshape((-1,) + pose.shape[-2:]),
            skeleton_config["KINEMATIC_TREE"],
            np.array(skeleton_config["OFFSET"]),
            forward_indices=[1, 0],
        ).reshape(pose.shape[:-1] + (-1,))

        if direction_process == "midfwd":
            ## Center frame of a window is translated to center and rotated to x+
            fwd_qtn = np.zeros((len(yaw), 4))
            fwd_qtn[:, [-1, 0]] = get_angle2D(yaw / 2)
            fwd_qtn = np.repeat(fwd_qtn[:, None, :], window, axis=1)
            local_qtn[..., 0, :] = qtn.qmul_np(fwd_qtn, local_qtn[..., 0, :])

            if "root" in data_keys:
                root = qtn.qrot_np(fwd_qtn, root)

        data["x6d"] = qtn.quaternion_to_cont6d_np(local_qtn)

    # Get offsets scaled by segment lengths
    if "offsets" in data_keys:
        if use_default_offsets:
            data["offsets"] = OFFSETS_3D_PAIRR24M[:, None] * np.array(
                skeleton_config["OFFSET"], dtype=np.float32
            )
        else:
            data["offsets"] = get_segment_len(
                pose.reshape((-1,) + pose.shape[-2:]),
                skeleton_config["KINEMATIC_TREE"],
                np.array(skeleton_config["OFFSET"]),
            ).reshape(pose.shape)

    # Get root positions
    if "root" in data_keys:
        data["root"] = root
        frame_dim_inds = tuple(range(len(root.shape) - 1))
        print("Root Maxes: {}".format(root.max(axis=frame_dim_inds)))
        print("Root Mins: {}".format(root.min(axis=frame_dim_inds)))

    data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}

    if "target_pose" in data_keys:
        # Target pose root does not move
        reshaped_x6d = data["x6d"].reshape((-1,) + data["x6d"].shape[-2:])
        if use_default_offsets:
            offsets = data["offsets"]
        else:
            offsets = data["offsets"].reshape(reshaped_x6d.shape[:2] + (-1,))
        data["target_pose"] = fwd_kin_cont6d_torch(
            reshaped_x6d,
            skeleton_config["KINEMATIC_TREE"],
            offsets,
            root_pos=torch.zeros(reshaped_x6d.shape[0], 3),
            do_root_R=True,
            eps=1e-8,
        ).reshape(data["x6d"].shape[:-1] + (3,))

    # if get_paired:
    #     data = {
    #         k: v.reshape((2, -1, *v.shape[1:])) if k != "offsets" else v
    #         for k, v in data.items()
    #     }
    data.update({k: torch.cat([v, v], dim=0) for k, v in labels_dict.items()})

    # Get animal IDs
    if "ids" in data_keys:
        data["ids"] = torch.tensor(ids, dtype=torch.int16)

    # import pdb; pdb.set_trace()

    for k, v in data.items():
        try:
            assert len(v) == data_len
        except:
            assert (len(v) == pose.shape[-2]) and (k == "offsets")

        print("{}: {}".format(k, v.shape))

    return data, t_paired_inds


class RatDatasetTPaired(Dataset):
    """
    Dataset class for Rat dataset
    """

    def __init__(
        self,
        data,
        paired_inds,
        arena_size=None,
        kinematic_tree=None,
        n_keypts=None,
        label="train",
        discrete_classes=None,
        norm_params=None,
    ):
        self.data_keys = list(data.keys())
        self.data = data
        self.n_keypts = n_keypts
        self.discrete_classes = discrete_classes
        self.norm_params = norm_params
        self.paired_inds = paired_inds

        if arena_size is not None:
            self.arena_size = torch.tensor(arena_size)
        else:
            self.arena_size = None

        self.kinematic_tree = kinematic_tree

        self.standard_offsets = False
        if "offsets" in self.data_keys:
            if self.data["offsets"].ndim == 2:
                self.standard_offsets = True

        # # List of items which have already been windowed
        # self.ind_with_window_inds = [
        #     k for k, v in self.data.items() if v.shape[0] != len(self.window_inds)
        # ]
        self.label = label

    def __len__(self):
        if self.get_social:
            return len(self.data[self.data_keys[1]]) * 2
        else:
            return len(self.data[self.data_keys[0]])

    def __getitem__(self, idx):
        # Use window indices to access arrays which have not been windowed
        # query = {
        #     k: self.data[k][self.window_inds[idx]] for k in self.ind_with_window_inds
        # }

        # if self.get_social:
        #     idx_a1 = int(idx >= len(self) // 2)
        #     idx_a2 = 1 - idx_a1
        #     query = {}
        #     for k, v in self.data.items():
        #         if (k in ["offsets"]) and self.standard_offsets:
        #             query[k] = v
        #         else:
        #             query[k] = v[[idx_a1, idx_a2], idx % (len(self) // 2)]
        # else:
        # Query items which have already been windowed
        query = {}
        for k, v in self.data.items():
            if k in ["target_pose", "pd_label", "ids"]:
                query[k] = v[self.paired_inds[idx, 1]]
            elif k in ["offsets"]:
                if self.standard_offsets:
                    continue
                else:
                    query[k] = v[self.paired_inds[idx, :]]
            else:
                query[k] = v[self.paired_inds[idx, :]]

        # query = {
        #     k: v[idx]
        #     for k, v in self.data.items()
        # }
        return query


class RatDataset(Dataset):
    """
    Dataset class for Rat dataset
    """

    def __init__(
        self,
        data,
        arena_size=None,
        kinematic_tree=None,
        n_keypts=None,
        label="train",
        discrete_classes=None,
        norm_params=None,
        get_social=False,
    ):
        self.data_keys = list(data.keys())
        self.data = data
        self.n_keypts = n_keypts
        self.discrete_classes = discrete_classes
        self.norm_params = norm_params
        self.get_social = get_social  # First dim is animal dimension, second dim is

        if arena_size is not None:
            self.arena_size = torch.tensor(arena_size)
        else:
            self.arena_size = None

        self.kinematic_tree = kinematic_tree

        self.standard_offsets = False
        if "offsets" in self.data_keys:
            if self.data["offsets"].ndim == 2:
                self.standard_offsets = True

        # # List of items which have already been windowed
        # self.ind_with_window_inds = [
        #     k for k, v in self.data.items() if v.shape[0] != len(self.window_inds)
        # ]
        self.label = label

    def __len__(self):
        if self.get_social:
            return len(self.data[self.data_keys[1]]) * 2
        else:
            return len(self.data[self.data_keys[0]])

    def __getitem__(self, idx):
        # Use window indices to access arrays which have not been windowed
        # query = {
        #     k: self.data[k][self.window_inds[idx]] for k in self.ind_with_window_inds
        # }

        if self.get_social:
            idx_a1 = int(idx >= len(self) // 2)
            idx_a2 = 1 - idx_a1
            query = {}
            for k, v in self.data.items():
                if (k in ["offsets"]) and self.standard_offsets:
                    query[k] = v
                else:
                    query[k] = v[[idx_a1, idx_a2], idx % (len(self) // 2)]
        else:
            # Query items which have already been windowed
            query = {}
            for k, v in self.data.items():
                if (k in ["offsets"]) and self.standard_offsets:
                    query[k] = v
                else:
                    query[k] = v[idx]

        # query = {
        #     k: v[idx]
        #     for k, v in self.data.items()
        # }
        return query


class RodentDiskDataset(Dataset):
    """
    Memory-mapped HDF5 dataset returning one model_window-frame window per
    sample for a single dataset.

    Adapted from befound's RodentDiskDataset (disk-backed loading). No
    preprocessing (centering, rotation, inverse kinematics) happens here --
    raw poses are returned and all of that is left to
    neurobe4.data.train_utils.prepare_batch.

    Each window is the middle model_window frames of a (~2s, native fps)
    window whose center is stored in the split's "_2s" index file.

    Example:
        dataset = RodentDiskDataset(
            data_path='/path/to/data/',
            dataset_name='wu_iclr25',
            train_val_test='train',
            stride=10,
        )
        dataset[0]["pose"].shape  # (model_window, n_keypts, 3)
    """

    def __init__(
        self,
        data_path: str,
        dataset_name: str,
        train_val_test: str,
        stride: int = 1,
        use_default_seg_len: bool = True,
        model_window: int = 51,
        virtual_rodent_x_splits: List[str] = ["base"],
        use_filtered_inds: bool = False,
        pose_dtype: torch.dtype = torch.float32,
    ):
        """
        Parameters
        ----------
        data_path : str
            Root directory containing one subdirectory per dataset name.
        dataset_name : str
            Dataset to load (e.g. "sdannce", "pairr24m", "wu_iclr25", ...).
            Social datasets ("sdannce", "pairr24m") are backed by
            pose_a1.h5/pose_a2.h5 and pose_single.h5, all loaded as samples.
            "parkinsons_healthy" reads wu_iclr25 and keeps only its baseline
            (pre-lesion) sessions, ids < 36.
        train_val_test : str
            Split to load ("train", "val", or "test").
        stride : int
            Stride over valid windows.
        use_default_seg_len : bool
            Load the canonical segment lengths (OFFSETS_3D) for
            get_default_offsets.
        model_window : int
            Number of frames returned per window, taken from the middle of
            each (~2s, native fps) window in the "_2s" index files.
        pose_dtype : torch.dtype
            dtype "pose" is cast to in __getitem__.
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.label = train_val_test
        self.model_window = model_window
        self.virtual_rodent_x_splits = virtual_rodent_x_splits
        self.pose_dtype = pose_dtype
        self.use_filtered_inds = use_filtered_inds
        self.stride = stride
        # Optional per-sample latents (see attach_latents)
        self.latents = None
        self.load_pose = True

        name = "wu_iclr25" if dataset_name == "parkinsons_healthy" else dataset_name
        self.h5_paths = self._get_h5_paths(data_path, name)

        # Skeleton config: kinematic tree, offset directions
        skeleton_config = read.config(f"{data_path}{name}/skeleton.yaml")
        self.kinematic_tree = skeleton_config["KINEMATIC_TREE"]
        self.skeleton_offsets = torch.tensor(
            skeleton_config["OFFSET"], dtype=torch.float32
        )
        if use_default_seg_len:
            self.default_seg_len = torch.tensor(OFFSETS_3D[name], dtype=torch.float32)
        else:
            self.default_seg_len = None

        # Open all H5 files and cache memory-mapped references
        self.h5_file_handles = {}
        self.h5_datasets = {}
        self.h5_ids = {}
        for h5_key, path in self.h5_paths.items():
            try:
                h5_file = h5py.File(path, "r")
            except Exception as e:
                self.close()
                raise RuntimeError(f"Failed to open H5 file {path}: {e}")
            self.h5_file_handles[h5_key] = h5_file
            self.h5_datasets[h5_key] = h5_file["pose"]
            # Per-frame session ids, in memory (small) so samples don't need a
            # second random disk read
            self.h5_ids[h5_key] = np.asarray(h5_file["ids"])
        self.n_keypts = self.h5_datasets[h5_key].shape[1]

        # Load window center indices per H5 file (frame indices local to that
        # file). self.centers holds every window of the split; self.indices
        # the valid (start) windows that define samples.
        self.centers = {}
        self.indices = {}
        unique_ids = []
        for h5_key in self.h5_paths.keys():
            centers = np.load(self._indices_path(data_path, h5_key))
            if centers.ndim == 2:  # legacy full-window format
                centers = centers[:, centers.shape[-1] // 2]

            ids = self.h5_ids[h5_key]
            centers = np.sort(centers).astype(np.int64)
            if dataset_name == "parkinsons_healthy":
                # Baseline sessions only
                centers = centers[ids[centers] < 36]
            self.centers[h5_key] = centers
            starts = self._get_valid_starts(centers, ids)
            self.indices[h5_key] = starts[::stride]
            unique_ids += [np.unique(ids[self.indices[h5_key]])]
            print(
                "Loading {} indices for {}: {} of {} windows".format(
                    train_val_test, h5_key, len(self.indices[h5_key]), len(centers)
                )
            )
        # Session ids present in this split (e.g. for discrete_classes["ids"])
        self.unique_ids = np.unique(np.concatenate(unique_ids))

        # Flatten per-file windows into one global sample list of
        # (h5_key, local_window_idx)
        self._entries = [
            (h5_key, i)
            for h5_key, inds in self.indices.items()
            for i in range(len(inds))
        ]

        # Row of each h5 file's first window in per-window arrays (e.g. cached
        # latents), which list self.centers of every h5 file in order
        n_centers = [len(c) for c in self.centers.values()]
        self.window_offsets = dict(
            zip(self.centers.keys(), np.cumsum([0] + n_centers[:-1]).tolist())
        )
        self.n_windows = sum(n_centers)

    def _get_h5_paths(self, data_path: str, name: str) -> Dict[str, str]:
        """{h5_key -> path}; some datasets are split across multiple h5 files"""
        if name in SOCIAL_DATASETS:
            return {
                f"{name}_{s}": f"{data_path}{name}/pose_{s}.h5"
                for s in ["a1", "a2", "single"]
            }
        elif name == "wu_iclr25":
            return {name: f"{data_path}{name}/parkinsons/pose.h5"}
        elif name == "virtual_rodent":
            return {
                f"{name}_{x}": f"{data_path}{name}/pose_{x}.h5"
                for x in self.virtual_rodent_x_splits
            }
        return {name: f"{data_path}{name}/pose.h5"}

    def _indices_path(self, data_path: str, h5_key: str) -> str:
        """Path to the "_2s" window-center split file for an h5 key"""
        name = self.dataset_name
        split = self.label
        filtered = FILTERED_SUFFIX if self.use_filtered_inds else ""
        if name in SOCIAL_DATASETS:
            # a1/a2 share one split, saved with a "_social" suffix
            kind = "single" if h5_key.endswith("_single") else "social"
            return f"{data_path}{name}/{split}_inds_{kind}_2s{filtered}.npy"
        elif name in ("wu_iclr25", "parkinsons_healthy"):
            return f"{data_path}wu_iclr25/parkinsons/{split}/inds_2s{filtered}.npy"
        elif name == "virtual_rodent":
            x_split = h5_key[len(name) + 1 :]
            x_split = "" if x_split == "base" else "_" + x_split
            return f"{data_path}{name}/{split}_inds{x_split}_2s{filtered}.npy"
        else:
            return f"{data_path}{name}/{split}_inds_2s{filtered}.npy"

    def _get_valid_starts(self, centers: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Every window center in the index file is a valid sample"""
        return centers

    def __len__(self) -> int:
        """Return number of samples across all h5 files"""
        return len(self._entries)

    def get_default_offsets(self, device) -> Optional[torch.Tensor]:
        """(n_keypts, 3) canonical offsets (segment length x direction).
        None if use_default_seg_len=False."""
        if self.default_seg_len is None:
            return None
        return (self.default_seg_len[:, None] * self.skeleton_offsets).to(device)

    def cache_name(self) -> str:
        """Name identifying this split's windows (self.centers) and their
        order, for per-window caches (e.g. latents). Independent of stride and
        time-pairing, which only choose which windows a sample uses. Carries
        the index file's filtering suffix, so caches of different filterings
        never share a name."""
        filtered = FILTERED_SUFFIX if self.use_filtered_inds else ""
        return "{}_{}_w{}{}".format(
            self.dataset_name, self.label, self.model_window, filtered
        )

    def window_rows(self, h5_key: str, centers: np.ndarray) -> np.ndarray:
        """Rows of the windows with these centers in per-window arrays"""
        return self.window_offsets[h5_key] + np.searchsorted(
            self.centers[h5_key], centers
        )

    def attach_latents(self, latents: np.ndarray, load_pose: bool = True):
        """
        Return the latents of each sample's window(s) under "vae_z".

        Parameters
        ----------
        latents : np.ndarray, (self.n_windows, z_dim)
            One latent per window of the split, ordered as self.centers of
            each h5 file in turn (may be a memmap).
        load_pose : bool
            If False, skip reading poses from disk (samples then only contain
            "vae_z", "ids" and, if applicable, "tau").
        """
        assert len(latents) == self.n_windows, (
            f"{len(latents)} latents for {self.n_windows} windows"
        )
        self.latents = latents
        self.load_pose = load_pose

    def _get_latents(self, h5_key: str, centers: np.ndarray) -> torch.Tensor:
        rows = self.window_rows(h5_key, centers)
        return torch.from_numpy(np.array(self.latents[rows], dtype=np.float32))

    def __getitem__(self, idx: int) -> dict:
        """
        Returns
        -------
        dict with keys:
            pose : torch.Tensor, shape (model_window, n_keypts, 3), self.pose_dtype
                (unless load_pose=False)
            ids : torch.Tensor, scalar, int16
            vae_z : torch.Tensor, (z_dim,), if latents are attached
        """
        h5_key, local_window_idx = self._entries[idx]
        center = int(self.indices[h5_key][local_window_idx])
        data = {
            "ids": torch.from_numpy(np.asarray(self.h5_ids[h5_key][center])).to(
                torch.int16
            ),
        }
        if self.load_pose:
            start = center - self.model_window // 2
            pose = self.h5_datasets[h5_key][start : start + self.model_window]
            data["pose"] = torch.from_numpy(pose).to(self.pose_dtype)
        if self.latents is not None:
            data["vae_z"] = self._get_latents(h5_key, np.array([center]))[0]
        return data

    def close(self):
        """Close all H5 files"""
        for h5_file in getattr(self, "h5_file_handles", {}).values():
            try:
                h5_file.close()
            except Exception:
                pass

    def __del__(self):
        self.close()


class RodentDiskDatasetTPaired(RodentDiskDataset):
    """
    RodentDiskDataset returning n_future_windows + 1 windows per sample, with
    consecutive window centers separated by n_skip frames.

    Merges RodentDiskDataset with MouseDatasetTPaired / preprocess_time_paired
    (time-paired indexing).

    For a start center c, the windows of one sample are centered at
        c, c + n_skip, ..., c + n_future_windows * n_skip
    and a start is only kept if every one of those centers is itself a valid
    window center in the split's index file (so speed/NaN-filtered windows are
    excluded, like the outlier filtering in preprocess_time_paired) and all
    centers belong to the same session id.

    With random_tau=True, each sample instead returns only the first window
    and one window tau * n_skip frames ahead, with tau drawn uniformly from
    {1, ..., n_future_windows} on every __getitem__ call.

    Example:
        dataset = RodentDiskDatasetTPaired(
            data_path='/path/to/data/',
            dataset_name='wu_iclr25',
            train_val_test='train',
            n_skip=25,
            n_future_windows=1,
            stride=10,
        )
        dataset[0]["pose"].shape  # (n_future_windows + 1, window, n_keypts, 3)
    """

    def __init__(
        self,
        data_path: str,
        dataset_name: str,
        train_val_test: str,
        n_skip: int = 25,
        n_future_windows: int = 1,
        require_contiguous: bool = False,
        random_tau: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_skip : int
            Frames (at the dataset's native fps) between consecutive window centers.
        n_future_windows : int
            Number of future windows; each sample has n_future_windows + 1 windows.
        require_contiguous : bool
            If True, every window center between the first and last window must
            also be valid (no filtered-out windows or split boundaries in
            between), mirroring preprocess_time_paired's per-section pairing.
            If False, only the n_future_windows + 1 window centers must be valid.
        random_tau : bool
            If True, return 2 windows per sample -- the first and one random
            tau in {1, ..., n_future_windows} steps ahead -- instead of all
            n_future_windows + 1. The sampled tau is returned under "tau".
        **kwargs
            Passed to RodentDiskDataset. stride is applied over valid *start*
            windows, after validity filtering, so future windows are always
            checked against the full index set.
        """
        # Needed by _get_valid_starts, which runs inside super().__init__
        self.n_skip = n_skip
        self.n_future_windows = n_future_windows
        self.require_contiguous = require_contiguous
        self.random_tau = random_tau
        super().__init__(data_path, dataset_name, train_val_test, **kwargs)

    def _get_valid_starts(self, centers: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """
        Return the subset of sorted window `centers` from which all
        n_future_windows future windows (spaced by n_skip) are also valid
        centers within the same session.
        """
        span = self.n_future_windows * self.n_skip
        if self.require_contiguous:
            # Last center of the consecutive run that each center belongs to
            breaks = np.flatnonzero(np.diff(centers) != 1) + 1
            run_idx = np.zeros(len(centers), dtype=np.int64)
            run_idx[breaks] = 1
            run_idx = np.cumsum(run_idx)
            run_end = centers[np.append(breaks - 1, len(centers) - 1)][run_idx]
            valid = centers + span <= run_end
        else:
            valid = np.ones(len(centers), dtype=bool)
            for t in range(1, self.n_future_windows + 1):
                future = centers + t * self.n_skip
                pos = np.clip(np.searchsorted(centers, future), 0, len(centers) - 1)
                valid &= centers[pos] == future

        # All windows must come from the same session
        starts = centers[valid]
        return starts[ids[starts] == ids[starts + span]]

    def __getitem__(self, idx: int) -> dict:
        """
        Returns
        -------
        dict with keys:
            pose : torch.Tensor, shape
                (n_future_windows + 1, model_window, n_keypts, 3), self.pose_dtype
                Window t is centered t * n_skip frames after window 0.
                If random_tau, shape (2, model_window, n_keypts, 3) with
                window 1 centered tau * n_skip frames after window 0.
                (unless load_pose=False)
            ids : torch.Tensor, scalar, int16 (shared by all windows)
            tau : torch.Tensor, scalar, int64, only if random_tau
            vae_z : torch.Tensor, (n_future_windows + 1, z_dim) -- or (2, z_dim)
                for windows [0, tau] if random_tau -- if latents are attached
        """
        h5_key, local_window_idx = self._entries[idx]
        start = int(self.indices[h5_key][local_window_idx])
        data = {
            "ids": torch.from_numpy(np.asarray(self.h5_ids[h5_key][start])).to(
                torch.int16
            ),
        }
        steps = self._sample_steps(data)
        data.update(self._read_windows(h5_key, start, steps))
        return data

    def _sample_steps(self, data: dict) -> np.ndarray:
        """Window steps of one sample, [0, tau] if random_tau (tau added to
        data) or 0..n_future_windows"""
        if not self.random_tau:
            return np.arange(self.n_future_windows + 1)
        # torch RNG so each DataLoader worker draws different taus
        tau = int(torch.randint(1, self.n_future_windows + 1, ()))
        data["tau"] = torch.tensor(tau, dtype=torch.long)
        return np.array([0, tau])

    def _read_windows(self, h5_key: str, start: int, steps: np.ndarray) -> dict:
        """{"vae_z", "pose"} (as enabled) of the windows centered at
        start + steps * n_skip in one h5 file"""
        data = {}
        if self.latents is not None:
            data["vae_z"] = self._get_latents(h5_key, start + steps * self.n_skip)
        if not self.load_pose:
            return data

        span_start = start - self.model_window // 2
        if len(steps) == 2:
            pose = np.stack(
                [
                    self.h5_datasets[h5_key][s : s + self.model_window]
                    for s in span_start + steps * self.n_skip
                ]
            )
        else:
            # Read the full contiguous span once (windows overlap when
            # n_skip < model_window, which h5py fancy indexing can't handle)
            span_end = span_start + steps[-1] * self.n_skip + self.model_window
            span = self.h5_datasets[h5_key][span_start:span_end]
            local_inds = (
                steps[:, None] * self.n_skip + np.arange(self.model_window)[None, :]
            )  # (len(steps), model_window)
            pose = span[local_inds]

        data["pose"] = torch.from_numpy(pose).to(self.pose_dtype)
        return data


class SocialDiskDatasetTPaired(RodentDiskDatasetTPaired):
    """
    RodentDiskDatasetTPaired over the social sessions of a paired dataset
    (SOCIAL_DATASETS: time-aligned pose_a1.h5 / pose_a2.h5; pose_single.h5 is
    not loaded), returning both animals' windows at the same frames.

    Animals are interchangeable: every time point is a sample once from each
    animal's perspective (target = a1, partner = a2, and vice versa), so the
    dataset has twice as many samples as social time points. Sample i and
    i + len(self) // 2 are the same time point.

    Example:
        dataset = SocialDiskDatasetTPaired(
            data_path='/path/to/data/',
            dataset_name='sdannce',
            train_val_test='train',
            n_skip=50,
            n_future_windows=10,
            stride=10,
        )
        sample = dataset[0]
        sample["pose"].shape, sample["pose_partner"].shape  # (11, window, n_keypts, 3)
    """

    def __init__(self, data_path: str, dataset_name: str, train_val_test: str, **kwargs):
        if dataset_name not in SOCIAL_DATASETS:
            raise ValueError(
                f"SocialDiskDatasetTPaired needs one of {SOCIAL_DATASETS}, got {dataset_name}"
            )
        if kwargs.get("random_tau"):
            # Social conditioning (rel_pose) uses the frame shared by windows
            # 0 and 1, so the target is always window 1
            raise ValueError("SocialDiskDatasetTPaired does not support random_tau")
        super().__init__(data_path, dataset_name, train_val_test, **kwargs)
        a1, a2 = (f"{dataset_name}_{a}" for a in ("a1", "a2"))
        # Shared split file and session ids, so both animals have the same windows
        assert np.array_equal(self.centers[a1], self.centers[a2])
        assert np.array_equal(self.h5_ids[a1], self.h5_ids[a2])
        self.partner_key = {a1: a2, a2: a1}
        self.target_index = {a1: 0, a2: 1}

    def _get_h5_paths(self, data_path: str, name: str) -> Dict[str, str]:
        """Social sessions only"""
        return {f"{name}_{a}": f"{data_path}{name}/pose_{a}.h5" for a in ("a1", "a2")}

    def cache_name(self) -> str:
        """Distinct from the full dataset's (which also lists pose_single windows)"""
        return super().cache_name().replace(
            self.dataset_name, self.dataset_name + "_social", 1
        )

    def __getitem__(self, idx: int) -> dict:
        """
        Returns
        -------
        dict with the keys of RodentDiskDatasetTPaired for the target animal
        (pose, ids, tau, vae_z), plus:
            pose_partner : torch.Tensor, like pose, the partner's windows at the
                same frames (unless load_pose=False)
            vae_z_partner : torch.Tensor, like vae_z, if latents are attached
            target : torch.Tensor, scalar, int64 -- 0 if the target is a1, 1 if a2
        """
        h5_key, local_window_idx = self._entries[idx]
        start = int(self.indices[h5_key][local_window_idx])
        data = {
            "ids": torch.from_numpy(np.asarray(self.h5_ids[h5_key][start])).to(
                torch.int16
            ),
            "target": torch.tensor(self.target_index[h5_key], dtype=torch.long),
        }
        steps = self._sample_steps(data)
        data.update(self._read_windows(h5_key, start, steps))
        partner = self._read_windows(self.partner_key[h5_key], start, steps)
        data.update({f"{k}_partner": v for k, v in partner.items()})
        return data


def collate_h5_dataset(batch: List[Dict]) -> Dict:
    """
    Collate function for RodentDiskDatasetTPaired batches. Stacks with
    torch.stack instead of the default collate_fn's num_workers>0 path, which
    resizes shared-memory storage and fails on tensors created via
    torch.from_numpy.
    """
    collated = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        collated[key] = (
            torch.stack(values) if isinstance(values[0], torch.Tensor) else values
        )
    return collated
