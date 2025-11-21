import scrubvae
import re
from pathlib import Path
import numpy as np
import torch
from neuroposelib.io import read
from typing import List, Optional
from scrubvae.data import quaternion as qtn
from scrubvae.data.dataset import (
    inv_kin,
    get_window_indices,
    get_speed_outliers,
    get_speed_parts,
    get_segment_len,
    get_angle2D,
    get_frame_yaw,
    fwd_kin_cont6d_torch,
    MouseDataset,
)
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import h5py


def get_data_and_model(
    config,
    load_model=None,
    epoch=None,
    train_val_test=["train", "val", "test"],
    data_keys=["x6d", "root", "offsets"],
    shuffle=[False, False, False],
    use_default_val_keys=True,
    verbose=1,
    n_time_steps=1,
    demo=False,
):
    if use_default_val_keys:
        if config["data"]["dataset"] == "4_mice":
            val_data_keys = [
                "ids",
                "x6d",
                "root",
                "offsets",
                "target_pose",
            ]
        else:
            val_data_keys = [
                "ids",
                "x6d",
                "root",
                "offsets",
                "target_pose",
                "pd_label",
            ]
    else:
        val_data_keys = data_keys

    if epoch is None:
        epoch = config["model"]["start_epoch"]

    if load_model is None:
        load_model = config["model"]["load_model"]

    ### Load Dataset
    # if train_val_test == "all":
    # next_window_len = 1 if "mlp" in config["model"]["type"] else config["model"]["window"]
    loader_dict = {}
    for is_shuffle, dataset_label in zip(shuffle, train_val_test):
        curr_data_keys = val_data_keys if dataset_label == "val" else data_keys
        loader_dict[dataset_label] = get_mouse_data(
            data_config=config["data"],
            train_val_test=dataset_label,
            data_keys=curr_data_keys,
            shuffle=is_shuffle,
            window=config["model"]["window"],
            n_time_steps=n_time_steps,
            demo=demo,
        )

    model = get_model(
        model_config=config["model"],
        load_model=load_model,
        epoch=epoch,
        disentangle_config=config["disentangle"],
        n_keypts=loader_dict[train_val_test[0]].dataset.n_keypts,
        direction_process=config["data"]["direction_process"],
        loss_config=config["loss"],
        arena_size=loader_dict[train_val_test[0]].dataset.arena_size,
        kinematic_tree=loader_dict[train_val_test[0]].dataset.kinematic_tree,
        bound=config["data"]["normalize"] == "bounded",
        discrete_classes=loader_dict[train_val_test[0]].dataset.discrete_classes,
        device="cuda",
        verbose=verbose,
    )

    return loader_dict, model


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

    Returns
    -------
    data
        Dictionary with key-value pairs associated with `data_keys`
    """
    n_ids = 72 if dataset == "parkinsons" else 4
    n_stride = 10 if dataset == "parkinsons" else 1
    print("Calculating dataset: {}".format(dataset))
    # Already windowed and split by dataset
    pose = np.load("{}{}/{}/pose.npy".format(data_path, dataset, train_val_test))
    ids = np.repeat(np.arange(n_ids), len(pose) // n_ids)
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
    data = {"raw_pose": pose}
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
        assert len(v) == data_len

    # import pdb; pdb.set_trace()
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

    def __len__(self):
        return len(self.paired_inds)

    def __getitem__(self, idx):
        # Query items which have already been windowed
        query = {}
        for k, v in self.data.items():
            if k in ["target_pose", "pd_label", "ids"]:
                query[k] = v[self.paired_inds[idx, 1]]
            else:
                query[k] = v[self.paired_inds[idx, :]]

        # query = {
        #     k: v[self.paired_inds[idx,:]]
        #     for k, v in self.data.items() if k != "target_pose" else v[self.paired_inds[idx,1]]
        # }
        return query


def get_mouse_data(
    data_config: dict,
    train_val_test: str = "train",
    data_keys: List[str] = ["x6d", "root", "offsets"],
    shuffle: bool = False,
    window: int = 51,
    n_time_steps: int = 1,
    demo: bool = False,
):
    """
    Load in mouse data and return pytorch dataloaders
    """
    skeleton_config = read.config(
        "{}mouse_skeleton.yaml".format(data_config["data_path"])
    )

    if window==1:
        data = scrubvae.data.dataset.preprocess_save_data(
                data_path=data_config["data_path"],
                skeleton_config=skeleton_config,
                dataset=data_config["dataset"],
                window=window,
                train_val_test=train_val_test,
                stride=1,
                data_keys=data_keys + ["ids"],
                speed_threshold=None if window == 1 else 2.5,
                direction_process=data_config["direction_process"],
            )
        paired_inds = None
    elif n_time_steps > 0:
        data, paired_inds = preprocess_time_paired(
            data_path=data_config["data_path"],
            skeleton_config=skeleton_config,
            dataset=data_config["dataset"],
            n_skip=data_config["time_steps"],
            train_val_test=train_val_test,
            window=window,
            data_keys=data_keys + ["ids"],
            speed_threshold=2.5,
            direction_process=data_config["direction_process"],
            n_time_steps=n_time_steps,
            demo=demo,
        )
    elif window==51 and n_time_steps==0:
        data_path = "{}{}/{}/".format(
            data_config["data_path"], data_config["dataset"], train_val_test
        )
        data = {}
        for key in data_keys + ["ids"]:
            if key in ["pd_label", "fluorescence"]:
                continue
            elif key in ["ids", "heading", "avg_speed_3d", "offsets", "raw_pose"]:
                file_path = "{}{}.h5".format(data_path, key)
            else:
                file_path = "{}{}_{}.h5".format(
                    data_path, key, data_config["direction_process"]
                )
            print("Reading in {} from {}".format(key, file_path))
            hf = h5py.File(file_path, "r")
            data[key] = np.array(hf.get(key))
            hf.close()
        data = {k: torch.from_numpy(v) for k, v in data.items()}
        paired_inds = None

    norm_params = {
        "avg_speed_3d": {
            "mean": torch.tensor([0.4993, 0.7112, 0.6663], dtype=torch.float32),
            "std": torch.tensor([0.4038, 0.3586, 0.4169], dtype=torch.float32),
        }
    }
    if "avg_speed_3d" in data_keys:
        print("Mean centering and unit standard deviation-scaling avg_speed_3d")
        data["avg_speed_3d"] -= norm_params["avg_speed_3d"]["mean"]
        data["avg_speed_3d"] /= norm_params["avg_speed_3d"]["std"]

        print("Speed Mins and Maxes:")
        print(data["avg_speed_3d"].min(dim=0)[0])
        print(data["avg_speed_3d"].max(dim=0)[0])

    discrete_classes = {}
    if data_config["dataset"] == "parkinsons":
        # Only if read in raw poses for the PD dataset
        if "pd_label" in data_keys:
            data["pd_label"] = torch.zeros((len(data["ids"]), 1)).long()
            data["pd_label"][data["ids"] >= 36] = 1
            discrete_classes["pd_label"] = [
                0,
                1,
            ]  # torch.unique(data["pd_label"], sorted=True)

        if "fluorescence" in data_keys:
            meta = pd.read_csv(
                data_config["data_path"] + data_config["dataset"] + "/metadata.csv"
            )
            # import pdb; pdb.set_trace()
            meta_by_frame = meta.iloc[data["ids"]]
            fluorescence = meta_by_frame["Fluorescence"].to_numpy()
            data["fluorescence"] = torch.tensor(fluorescence, dtype=torch.float32)

        data["ids"][data["ids"] >= 36] = data["ids"][data["ids"] >= 36] - 36
        unique_ids = torch.unique(data["ids"])
        discrete_classes["ids"] = torch.arange(len(unique_ids)).long()
    else:
        discrete_classes["ids"] = torch.unique(data["ids"], sorted=True)

    if paired_inds is not None:
        dataset = MouseDatasetTPaired(
            data=data,
            paired_inds=paired_inds,
            arena_size=data_config["arena_size"],
            kinematic_tree=skeleton_config["KINEMATIC_TREE"],
            n_keypts=len(skeleton_config["LABELS"]),
            label=train_val_test,
            discrete_classes=discrete_classes,
            norm_params=norm_params,
        )
    else:
        dataset = MouseDataset(
            data,
            data_config["arena_size"],
            skeleton_config["KINEMATIC_TREE"],
            len(skeleton_config["LABELS"]),
            label=train_val_test,
            discrete_classes=discrete_classes,
            norm_params=norm_params,
        )

    loader = DataLoader(
        dataset=dataset,
        batch_size=data_config["batch_size"],
        shuffle=shuffle,
        num_workers=5,
        pin_memory=True,
    )

    return loader


def get_model(
    model_config,
    load_model,
    epoch,
    disentangle_config,
    n_keypts,
    direction_process,
    loss_config=None,
    arena_size=None,
    kinematic_tree=None,
    bound=False,
    discrete_classes=None,
    device="cuda",
    verbose=1,
):
    feat_dim_dict = {
        "avg_speed": 1,
        "part_speed": 4,
        "frame_speed": model_config["window"] - 1,
        "avg_speed_3d": 3,
        "heading": 2,
        "heading_change": 1,
        "fluorescence": 1,
    }

    if discrete_classes is not None:
        print("Discrete Classes: {}".format(discrete_classes))
        feat_dim_dict.update({k: len(v) for k, v in discrete_classes.items()})

    in_channels = n_keypts * 6
    if direction_process in ["x360", "midfwd", None]:
        in_channels += 3

    methods = disentangle_config["method"]
    disentangle = {}
    # Linear Projection model for each disentanglement feature
    if "linear" in methods.keys():  # len(methods["linear"]) > 0:
        from scrubvae.model.disentangle import LinearProjection

        disentangle["linear"] = {}
        for feat in methods["linear"]:
            disentangle["linear"][feat] = LinearProjection(
                model_config["z_dim"],
                feat_dim_dict[feat],
                bias=False,
            )

    # Conditional VAE adding dimensions to decoder input
    if "conditional" in methods.keys():  # len(methods["conditional"]) > 0:
        conditional_dim = sum([feat_dim_dict[k] for k in methods["conditional"]])
        conditional_keys = methods["conditional"]
    else:
        conditional_keys = None
        conditional_dim = 0

    if len(model_config["conditional_keys"]) > 0:
        diffusion_conditional_dim = sum(
            [feat_dim_dict[k] for k in model_config["conditional_keys"]]
        )
    else:
        diffusion_conditional_dim = 0

    # Gradient reversal scrubbing for each disentanglement feature
    if "grad_reversal" in methods.keys():  # len(methods["grad_reversal"]) > 0:
        from scrubvae.model.disentangle import GRScrubber

        disentangle["grad_reversal"] = {}
        for feat in methods["grad_reversal"]:
            disentangle["grad_reversal"][feat] = GRScrubber(
                model_config["z_dim"],
                feat_dim_dict[feat],
                alpha=disentangle_config["alpha"],
                bound=bound,
            )

    # Moving Average Least Squares Filter with n-order polynomical features
    if "moving_avg_lsq" in methods.keys():  # len(methods["moving_avg_lsq"]) > 0:
        from scrubvae.model.disentangle import MovingAvgLeastSquares

        disentangle["moving_avg_lsq"] = {}
        for feat in methods["moving_avg_lsq"]:
            disentangle["moving_avg_lsq"][feat] = MovingAvgLeastSquares(
                model_config["z_dim"],
                feat_dim_dict[feat],
                bias=loss_config[feat + "_mals"] < 0,
                polynomial_order=disentangle_config["polynomial"],
                l2_reg=disentangle_config["l2_reg"],
            )

    # Quadratic Discriminant Filter for class scrubbing
    if "qda" in methods.keys():  # len(methods["qda"]) > 0:
        from scrubvae.model.disentangle import QuadraticDiscriminantFilter

        disentangle["qda"] = {}
        for feat in methods["qda"]:
            disentangle["qda"][feat] = QuadraticDiscriminantFilter(
                model_config["z_dim"], discrete_classes[feat]
            )

    # Moving Average Filter for class scrubbing
    if "moving_avg" in methods.keys():  # len(methods["moving_avg"]) > 0:
        from scrubvae.model.disentangle import MovingAverageFilter

        disentangle["moving_avg"] = {}
        for feat in methods["moving_avg"]:
            disentangle["moving_avg"][feat] = MovingAverageFilter(
                model_config["z_dim"], discrete_classes[feat]
            )

    if "adversarial_net" in methods.keys():
        from scrubvae.model.disentangle import AdvNetScrubber

        disentangle["adversarial_net"] = {}
        for feat in methods["adversarial_net"]:
            disentangle["adversarial_net"][feat] = AdvNetScrubber(
                model_config["z_dim"] + conditional_dim
            )

    ### Initialize/load model
    if model_config["type"] == "conditional_forecaster":
        print("Conditional Forecasting VAE")
        from scrubvae.model.residual import ConditionalForecaster

        vae = ConditionalForecaster(
            in_channels=in_channels,
            kernel=model_config["kernel"],
            z_dim=model_config["z_dim"],
            window=model_config["window"],
            activation=model_config["activation"],
            is_diag=model_config["diag"],
            conditional_dim=conditional_dim,
            init_dilation=model_config["init_dilation"],
            disentangle=disentangle,
            disentangle_keys=disentangle_config["features"],
            conditional_keys=conditional_keys,
            arena_size=arena_size,
            kinematic_tree=kinematic_tree,
            prior=model_config["prior"],
            ch=model_config["channel"],
            discrete_classes=discrete_classes,
        )
    elif model_config["type"] == "predictive_forecaster":
        print("Forecasting VAE")
        from scrubvae.model.residual import ForecastingVAE

        vae = ForecastingVAE(
            in_channels=in_channels,
            kernel=model_config["kernel"],
            z_dim=model_config["z_dim"],
            window=model_config["window"],
            activation=model_config["activation"],
            is_diag=model_config["diag"],
            conditional_dim=conditional_dim,
            init_dilation=model_config["init_dilation"],
            disentangle=disentangle,
            disentangle_keys=disentangle_config["features"],
            conditional_keys=conditional_keys,
            arena_size=arena_size,
            kinematic_tree=kinematic_tree,
            prior=model_config["prior"],
            ch=model_config["channel"],
            discrete_classes=discrete_classes,
        )
    elif model_config["type"] == "mlp":
        print("Single Pose VAE")
        from neurobe4.model.model import MLPVAE

        vae = MLPVAE(
            in_dim=in_channels,
            layer_dims=model_config["channel"],
            z_dim=model_config["z_dim"],
            activation=model_config["activation"],
            is_diag=model_config["diag"],
            prior=model_config["prior"],
            arena_size=arena_size,
            kinematic_tree=kinematic_tree,
        )
    elif model_config["type"] == "mlp_diff":
        from neurobe4.model.model import MLPDiff

        vae = MLPDiff(in_dim=in_channels,
            layer_dims=model_config["channel"],
            z_dim=model_config["z_dim"],
            activation=model_config["activation"],
            is_diag=model_config["diag"],
            prior=model_config["prior"],
            arena_size=arena_size,
            kinematic_tree=kinematic_tree,
            diffusion_cond_keys=model_config["conditional_keys"],
            diffusion_cond_dim=diffusion_conditional_dim,
            window=model_config["window"],
            cfg_prob=model_config["cfg_prob"],
        )

    # elif model_config["type"] == "conditional_diffusion":
    #     print("Forecasting VAE")
    #     from model import ConditionalDiffusion

    #     vae = ConditionalDiffusion(
    #         in_channels=in_channels,
    #         kernel=model_config["kernel"],
    #         z_dim=model_config["z_dim"],
    #         window=model_config["window"],
    #         activation=model_config["activation"],
    #         is_diag=model_config["diag"],
    #         conditional_dim=conditional_dim,
    #         init_dilation=model_config["init_dilation"],
    #         disentangle=disentangle,
    #         disentangle_keys=disentangle_config["features"],
    #         conditional_keys=conditional_keys,
    #         arena_size=arena_size,
    #         kinematic_tree=kinematic_tree,
    #         prior=model_config["prior"],
    #         ch=model_config["channel"],
    #         discrete_classes=discrete_classes,
    #     )
    elif model_config["type"] == "nc_conditional_diffusion":
        print("Forecasting VAE")
        from neurobe4.model.model import ConditionalDiffusion

        vae = ConditionalDiffusion(
            in_channels=in_channels,
            kernel=model_config["kernel"],
            z_dim=model_config["z_dim"],
            window=model_config["window"],
            activation=model_config["activation"],
            is_diag=model_config["diag"],
            conditional_dim=conditional_dim,
            init_dilation=model_config["init_dilation"],
            disentangle=disentangle,
            disentangle_keys=disentangle_config["features"],
            conditional_keys=conditional_keys,
            arena_size=arena_size,
            kinematic_tree=kinematic_tree,
            prior=model_config["prior"],
            ch=model_config["channel"],
            discrete_classes=discrete_classes,
            diffusion_cond_keys=model_config["conditional_keys"],
            diffusion_cond_dim=diffusion_conditional_dim,
            cfg_prob=model_config["cfg_prob"],
        )
    elif model_config["type"] == "diff_vae":
        print("Forecasting VAE")
        from neurobe4.model.model import DiffVAE

        vae = DiffVAE(
            in_channels=in_channels,
            kernel=model_config["kernel"],
            z_dim=model_config["z_dim"],
            window=model_config["window"],
            activation=model_config["activation"],
            is_diag=model_config["diag"],
            conditional_dim=conditional_dim,
            init_dilation=model_config["init_dilation"],
            disentangle=disentangle,
            disentangle_keys=disentangle_config["features"],
            conditional_keys=conditional_keys,
            arena_size=arena_size,
            kinematic_tree=kinematic_tree,
            prior=model_config["prior"],
            ch=model_config["channel"],
            discrete_classes=discrete_classes,
            diffusion_cond_keys=model_config["conditional_keys"],
            diffusion_cond_dim=diffusion_conditional_dim,
            cfg_prob=model_config["cfg_prob"],
        )

    if verbose > 0:
        print(vae)

    if load_model is not None:
        load_path = "{}/weights/epoch_{}.pth".format(load_model, epoch)
        print("Loading Weights from:\n{}".format(load_path))
        state_dict = torch.load(load_path)
        missing_keys, unexpected_keys = vae.load_state_dict(state_dict, strict=False)

        if verbose > 0:
            print("Missing Keys: {}".format(missing_keys))
            print("Unexpected Keys: {}".format(unexpected_keys))

    return vae.to(device)
