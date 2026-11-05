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
from neurobe4.data.dataset import (
    MouseDatasetTPaired,
    preprocess_time_paired,
    preprocess_pairr24m_data,
    preprocess_time_paired_pairr24m,
    RodentDiskDataset,
    RodentDiskDatasetTPaired,
    SocialDiskDatasetTPaired,
    collate_h5_dataset,
)
from neurobe4.data.latents import get_cached_latents, get_latent_norm


def get_data_and_model(
    config,
    load_model=None,
    epoch=None,
    train_val_test=["train", "val", "test"],
    shuffle=[False, False, False],
    verbose=1,
    device="cuda",
):
    if epoch is None:
        epoch = config["model"]["start_epoch"]

    if load_model is None:
        load_model = config["model"]["load_model"]

    ### Load Dataset
    loader_dict = {}
    for is_shuffle, dataset_label in zip(shuffle, train_val_test):
        loader_dict[dataset_label] = get_rodent_data_disk(
            data_config=config["data"],
            train_val_test=dataset_label,
            shuffle=is_shuffle,
            model_window=config["model"]["window"],
        )

    # Cached latents from the pretrained VAE at data.vae_z_path:
    # - diff_vae with a frozen autoencoder trains directly on them (no poses)
    # - conditional_diff with a frozen autoencoder uses them as the target
    #   latent; poses are still loaded for its trainable past_encoder
    # - latent_norm statistics are computed from the training set's latents
    use_vae_z = bool(
        config["data"]["vae_z_path"]
        and config["train"]["freeze_ae"]
        and config["model"]["type"] in ("diff_vae", "conditional_diff", "social_diff")
    )
    load_pose = config["model"]["type"] in ("conditional_diff", "social_diff")
    if config["model"]["type"] == "social_diff" and not config["data"].get("social"):
        raise ValueError("social_diff models need data.social: true (SocialDiskDatasetTPaired)")
    compute_latent_norm = (config["model"].get("latent_norm") is not None) and (
        "train" in train_val_test
    )
    latent_norm = None
    if use_vae_z or compute_latent_norm:
        assert config["data"]["vae_z_path"], "vae_z_path is required for cached latents"
        vae = get_latent_vae(
            config["data"], loader_dict[train_val_test[0]].dataset, device=device
        )
        for dataset_label, loader in loader_dict.items():
            if not (use_vae_z or dataset_label == "train"):
                continue
            latents = get_cached_latents(
                loader.dataset,
                vae,
                cache_path=get_latent_cache_path(config["data"], loader.dataset),
                batch_size=config["data"]["batch_size"],
                device=device,
            )
            if use_vae_z:
                loader.dataset.attach_latents(latents, load_pose=load_pose)
            if compute_latent_norm and dataset_label == "train":
                assert hasattr(
                    loader.dataset, "n_skip"
                ), "latent_norm requires time-paired data"
                latent_norm = get_latent_norm(
                    loader.dataset,
                    latents,
                    config["model"]["forecasting_type"],
                    config["model"]["latent_norm"],
                )
        del vae

    print("Latent Scaling: ", latent_norm)
    model = get_model(
        model_config=config["model"],
        load_model=load_model,
        epoch=epoch,
        disentangle_config=config["disentangle"],
        n_keypts=loader_dict[train_val_test[0]].dataset.n_keypts,
        direction_process=config["data"]["direction_process"],
        arena_size=loader_dict[train_val_test[0]].dataset.arena_size,
        kinematic_tree=loader_dict[train_val_test[0]].dataset.kinematic_tree,
        discrete_classes=loader_dict[train_val_test[0]].dataset.discrete_classes,
        latent_norm=latent_norm,
        device=device,
        verbose=verbose,
    )
    # import pdb; pdb.set_trace()

    return loader_dict, model


def get_mouse_data(
    data_config: dict,
    train_val_test: str = "train",
    data_keys: List[str] = ["x6d", "root", "offsets"],
    shuffle: bool = False,
    window: int = 51,
    n_time_steps: int = 1,
    demo: bool = False,
    get_social_paired: bool = False,
):
    """
    Load in mouse data and return pytorch dataloaders
    """
    if n_time_steps is None:
        n_time_steps = 0
    if data_config["dataset"] == "pairr24m":
        skeleton_config = read.config(
            "{}rat_skeleton.yaml".format(data_config["data_path"])
        )
    else:
        skeleton_config = read.config(
            "{}mouse_skeleton.yaml".format(data_config["data_path"])
        )

    if data_config["dataset"] == "pairr24m":
        if (window == 51 and n_time_steps in [0, None]) or (window == 1):
            data = preprocess_pairr24m_data(
                data_path=data_config["data_path"],
                skeleton_config=skeleton_config,
                dataset=data_config["dataset"],
                window=window,
                train_val_test=train_val_test,
                stride=1,
                data_keys=data_keys + ["ids"],
                use_default_offsets=True,
                direction_process=data_config["direction_process"],
                get_social_paired=get_social_paired,
                demo=demo,
            )
            paired_inds = None
        else:
            data, paired_inds = preprocess_time_paired_pairr24m(
                data_path=data_config["data_path"],
                skeleton_config=skeleton_config,
                dataset=data_config["dataset"],
                n_skip=data_config["n_skip"],
                train_val_test=train_val_test,
                window=window,
                data_keys=data_keys + ["ids"],
                direction_process=data_config["direction_process"],
                n_time_steps=n_time_steps,
                use_default_offsets=True,
                demo=demo,
            )

    else:
        if window == 1:
            data = scrubvae.data.dataset.preprocess_save_data(
                data_path=data_config["data_path"],
                skeleton_config=skeleton_config,
                dataset=data_config["dataset"],
                window=window,
                train_val_test=train_val_test,
                stride=1,
                data_keys=data_keys + ["ids"],
                speed_threshold=None if window == 1 else 2.5,
                use_default_offsets=True,
                direction_process=data_config["direction_process"],
            )
            paired_inds = None
        elif n_time_steps > 0:
            data, paired_inds = preprocess_time_paired(
                data_path=data_config["data_path"],
                skeleton_config=skeleton_config,
                dataset=data_config["dataset"],
                n_skip=data_config["n_skip"],
                train_val_test=train_val_test,
                window=window,
                data_keys=data_keys + ["ids"],
                speed_threshold=2.5,
                direction_process=data_config["direction_process"],
                n_time_steps=n_time_steps,
                use_default_offsets=True,
                demo=demo,
            )
        elif window == 51 and n_time_steps == 0:
            if data_config["dataset"] == "parkinsons_healthy":
                dataset_name = "parkinsons"
            else:
                dataset_name = data_config["dataset"]
            data_path = "{}{}/{}/".format(
                data_config["data_path"], dataset_name, train_val_test
            )
            data = {}
            for key in data_keys + ["ids"]:
                if key in ["pd_label", "fluorescence", "vae_z"]:
                    continue
                elif key in ["ids", "heading", "avg_speed_3d", "raw_pose"]:
                    file_path = "{}{}.h5".format(data_path, key)
                elif key == "offsets":
                    data["offsets"] = np.array(
                        [
                            0.0,
                            17.0,
                            14.0,
                            19.0,
                            24.0,
                            42.0,
                            23.5,
                            11.5,
                            3.0,
                            23.5,
                            11.5,
                            3.0,
                            29.5,
                            17.0,
                            11.0,
                            29.5,
                            17.0,
                            11.0,
                        ],
                        dtype=np.float32,
                    )
                    data["offsets"] = data["offsets"][:, None] * np.array(
                        skeleton_config["OFFSET"], dtype=np.float32
                    )
                    continue
                else:
                    file_path = "{}{}_{}.h5".format(
                        data_path, key, data_config["direction_process"]
                    )
                print("Reading in {} from {}".format(key, file_path))
                hf = h5py.File(file_path, "r")
                data[key] = np.array(hf.get(key))
                hf.close()

            if data_config["dataset"] == "parkinsons_healthy":
                healthy_inds = data["ids"] < 36
                data = {
                    k: v[healthy_inds] if k != "offsets" else v for k, v in data.items()
                }

            data = {
                k: v if torch.is_tensor(v) else torch.from_numpy(v)
                for k, v in data.items()
            }
            paired_inds = None

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

    if "vae_z" in data_keys:
        vae_z = np.load(
            data_config["vae_z_path"]
            + "/latents/gt_"
            + train_val_test
            + "_z_1steps.npy"
        )
        data["vae_z"] = torch.tensor(vae_z, dtype=torch.float32)

    if paired_inds is not None:
        dataset = MouseDatasetTPaired(
            data=data,
            paired_inds=paired_inds,
            arena_size=data_config["arena_size"],
            kinematic_tree=skeleton_config["KINEMATIC_TREE"],
            n_keypts=len(skeleton_config["LABELS"]),
            label=train_val_test,
            discrete_classes=discrete_classes,
        )
    else:
        dataset = MouseDataset(
            data,
            data_config["arena_size"],
            skeleton_config["KINEMATIC_TREE"],
            len(skeleton_config["LABELS"]),
            label=train_val_test,
            discrete_classes=discrete_classes,
            norm_params=None,
        )

    loader = DataLoader(
        dataset=dataset,
        batch_size=data_config["batch_size"],
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True,
    )

    return loader


def get_rodent_data_disk(
    data_config: dict,
    train_val_test: str = "train",
    shuffle: bool = False,
    model_window: int = 51,
):
    """
    Disk-backed (h5) loader returning raw poses; preprocessing happens per
    batch in neurobe4.data.train_utils.prepare_batch. Uses RodentDiskDatasetTPaired
    if data_config["n_future_windows"] > 0, otherwise RodentDiskDataset; with
    data_config["social"], SocialDiskDatasetTPaired (both animals of the
    social sessions; requires n_future_windows > 0).

    data_config["data_path"] is the root data directory (containing one
    folder per dataset) and data_config["dataset"] the dataset folder name
    (e.g. "wu_iclr25").
    """
    kwargs = dict(
        data_path=data_config["data_path"],
        dataset_name=data_config["dataset"],
        train_val_test=train_val_test,
        stride=data_config["stride"] or 1,
        model_window=model_window,
        use_filtered_inds=bool(data_config["use_filtered_inds"]),
    )
    if data_config.get("social") and not data_config["n_future_windows"]:
        raise ValueError("social datasets are time-paired: set n_future_windows > 0")
    if data_config["n_future_windows"]:
        dataset_class = (
            SocialDiskDatasetTPaired if data_config.get("social") else RodentDiskDatasetTPaired
        )
        dataset = dataset_class(
            n_skip=data_config["n_skip"],
            n_future_windows=data_config["n_future_windows"],
            require_contiguous=bool(data_config["require_contiguous"]),
            random_tau=bool(data_config["random_tau"]),
            **kwargs,
        )
    else:
        dataset = RodentDiskDataset(**kwargs)

    # Attributes get_data_and_model reads off the dataset to build the model
    dataset.arena_size = (
        torch.tensor(data_config["arena_size"])
        if data_config["arena_size"] is not None
        else None
    )
    dataset.discrete_classes = {"ids": torch.from_numpy(dataset.unique_ids)}

    loader = DataLoader(
        dataset=dataset,
        batch_size=data_config["batch_size"],
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_h5_dataset,
    )

    return loader


def get_latent_vae(data_config: dict, dataset, device: str = "cuda"):
    """
    Pretrained (scrubvae) VAE at data_config["vae_z_path"], loaded at epoch
    data_config["vae_z_epoch"] (default 600), used to encode cached latents.
    """
    v_config = read.config(data_config["vae_z_path"] + "/model_config.yaml")
    vae = scrubvae.get.model(
        model_config=v_config["model"],
        load_model=data_config["vae_z_path"],
        epoch=data_config["vae_z_epoch"] or 600,
        disentangle_config=v_config["disentangle"],
        n_keypts=dataset.n_keypts,
        direction_process=v_config["data"]["direction_process"],
        loss_config=v_config["loss"],
        arena_size=dataset.arena_size,
        kinematic_tree=dataset.kinematic_tree,
        bound=v_config["data"]["normalize"] == "bounded",
        device=device,
        verbose=-1,
    )
    return vae.eval()


def get_latent_cache_path(data_config: dict, dataset) -> str:
    """Cache file for `dataset`'s latents under {vae_z_path}/latents/"""
    return "{}/latents/{}_epoch{}.npy".format(
        data_config["vae_z_path"].rstrip("/"),
        dataset.cache_name(),
        data_config["vae_z_epoch"] or 600,
    )


def get_model(
    model_config,
    load_model,
    epoch,
    disentangle_config,
    n_keypts,
    direction_process,
    arena_size=None,
    kinematic_tree=None,
    latent_norm=None,
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

    # "tau" in conditional_keys conditions on the forecast horizon through its
    # own token (with its own guidance null); other keys are concatenated as
    # extra conditioning variables
    condition_tau = "tau" in (model_config["conditional_keys"] or [])
    diffusion_cond_keys = [k for k in (model_config["conditional_keys"] or []) if k != "tau"]
    if len(diffusion_cond_keys) > 0:
        diffusion_conditional_dim = sum(
            [feat_dim_dict[k] for k in diffusion_cond_keys]
        )
    else:
        diffusion_conditional_dim = 0

    ### Initialize/load model
    if model_config["type"] == "conditional_forecaster":
        print("Conditional Forecasting VAE")
        from neurobe4.model.model import ConditionalForecaster

        vae = ConditionalForecaster(
            in_channels=in_channels,
            kernel=model_config["kernel"],
            z_dim=model_config["z_dim"],
            window=model_config["window"],
            activation=model_config["activation"],
            is_diag=model_config["diag"],
            conditional_dim=0,
            init_dilation=model_config["init_dilation"],
            disentangle=disentangle,
            disentangle_keys=disentangle_config["features"],
            conditional_keys=None,
            arena_size=arena_size,
            kinematic_tree=kinematic_tree,
            prior=model_config["prior"],
            ch=model_config["channel"],
            discrete_classes=discrete_classes,
        )
    elif model_config["type"] == "predictive_forecaster":
        print("Forecasting VAE")
        from neurobe4.model.model import ForecastingVAE

        vae = ForecastingVAE(
            in_channels=in_channels,
            kernel=model_config["kernel"],
            z_dim=model_config["z_dim"],
            window=model_config["window"],
            activation=model_config["activation"],
            is_diag=model_config["diag"],
            conditional_dim=0,
            init_dilation=model_config["init_dilation"],
            disentangle=disentangle,
            disentangle_keys=disentangle_config["features"],
            conditional_keys=None,
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
    elif model_config["type"] in ("conditional_diff", "social_diff"):
        print("Forecasting VAE")
        from neurobe4.model.model import ConditionalDiffusion, SocialConditionalDiffusion

        if model_config.get("forecasting_type") == "velocity":
            # ConditionalDiffusion always predicts the future latent itself;
            # its z_past comes from a separate past_encoder, not the VAE
            # encoder, so mu_future - z_past would not be a velocity
            raise ValueError(
                "forecasting_type: velocity is not implemented for "
                "conditional_diff (it always uses the center target); "
                "remove it or set it to center"
            )

        social_kwargs = {}
        if model_config["type"] == "social_diff":
            # Sources and the subsets trained on (see SocialConditionalDiffusion)
            if model_config["social_sources"] is not None:
                social_kwargs["social_sources"] = model_config["social_sources"]
            social_kwargs["social_subsets"] = model_config["social_subsets"]
        model_class = (
            SocialConditionalDiffusion
            if model_config["type"] == "social_diff"
            else ConditionalDiffusion
        )
        vae = model_class(
            **social_kwargs,
            in_channels=in_channels,
            kernel=model_config["kernel"],
            z_dim=model_config["z_dim"],
            window=model_config["window"],
            activation=model_config["activation"],
            is_diag=model_config["diag"],
            conditional_dim=0,
            init_dilation=model_config["init_dilation"],
            disentangle=disentangle,
            disentangle_keys=disentangle_config["features"],
            conditional_keys=None,
            arena_size=arena_size,
            kinematic_tree=kinematic_tree,
            prior=model_config["prior"],
            ch=model_config["channel"],
            discrete_classes=discrete_classes,
            diffusion_cond_keys=diffusion_cond_keys,
            diffusion_cond_dim=diffusion_conditional_dim,
            cfg_prob=model_config["cfg_prob"],
            diffusion_type=model_config["diffusion_type"],
            latent_norm=latent_norm,
            # Only an explicit true keeps the bottleneck (null -> past tokens)
            bottleneck_past=model_config["bottleneck_past"] is True,
            expand_latent=model_config["expand_latent"],
            condition_tau=condition_tau,
            cfg_prob_tau=model_config["cfg_prob_tau"] or 0.0,
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
            conditional_dim=0,
            init_dilation=model_config["init_dilation"],
            disentangle=disentangle,
            disentangle_keys=disentangle_config["features"],
            conditional_keys=None,
            arena_size=arena_size,
            kinematic_tree=kinematic_tree,
            prior=model_config["prior"],
            ch=model_config["channel"],
            discrete_classes=discrete_classes,
            diffusion_cond_keys=diffusion_cond_keys,
            diffusion_cond_dim=diffusion_conditional_dim,
            cfg_prob=model_config["cfg_prob"],
            diffusion_type=model_config["diffusion_type"],
            forecasting_type=model_config.get("forecasting_type") or "velocity",
            latent_norm=latent_norm,
            expand_latent=model_config["expand_latent"],
            condition_tau=condition_tau,
            cfg_prob_tau=model_config["cfg_prob_tau"] or 0.0,
        )
    elif model_config["type"] == "rcnn":
        from scrubvae.model.residual import ResVAE

        vae = ResVAE(
            in_channels=in_channels,
            kernel=model_config["kernel"],
            z_dim=model_config["z_dim"],
            window=model_config["window"],
            activation=model_config["activation"],
            is_diag=model_config["diag"],
            conditional_dim=0,
            init_dilation=model_config["init_dilation"],
            disentangle=disentangle,
            disentangle_keys=disentangle_config["features"],
            conditional_keys=None,
            arena_size=arena_size,
            kinematic_tree=kinematic_tree,
            prior=model_config["prior"],
            ch=model_config["channel"],
            discrete_classes=discrete_classes,
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
