import scrubvae
from neuroposelib import read
import numpy as np
import torch
import tqdm
from neuroposelib.datastruct import Connectivity
from neurobe4 import get
import torch.nn.functional as F
from scrubvae.data import quaternion as qtn
from scrubvae.data.rotation_conversion import matrix_to_rotation_6d
from scrubvae.data.dataset import inv_kin_torch, fwd_kin_cont6d_torch
from neurobe4.model.model import ConditionalDiffusion, ForecastingVAE
from neurobe4.data.train_utils import prepare_batch

def generate_samples(
    model_path,
    epoch,
    train_val_test,
    n_future_windows,
    n_samples,
    n_skip: int=50,
    uncond_scale: int = 0,
    ddim_steps: int = 0,
):
    config = read.config(model_path + "model_config.yaml")
    config["model"]["load_model"] = config["out_path"]
    config["model"]["start_epoch"] = epoch
    config["data"]["n_skip"] = n_skip
    config["data"]["n_future_windows"] = n_future_windows
    # Fixed, consecutive windows (not a random horizon per sample), so
    # forecasts line up with ground truth (scripts/eval_common.py)
    config["data"]["random_tau"] = False

    loader_dict, model = get.get_data_and_model(
        config,
        train_val_test=[train_val_test],
        shuffle=[False],
    )
    loader = loader_dict[train_val_test]
    # Sampling encodes the past window and decodes poses, so it needs poses
    # even for models that train on cached latents alone (diff_vae with
    # vae_z_path), whose loaders skip pose reads
    loader.dataset.latents = None
    loader.dataset.load_pose = True
    print("Dataset Length: {}".format(len(loader.dataset)))
    model.eval()

    # data_o = {}
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm.tqdm(loader)):
            # in_time_steps = 2 if isinstance(model, ConditionalDiffusion) else 1
            data = prepare_batch(data, loader.dataset, device="cuda")
            data = {k: data[k] for k in ["x6d", "root", "offsets"]}

            data_batch = model.get_samples(
                data,
                n_future_windows=n_future_windows,
                n_samples=n_samples,
                uncond_scale=uncond_scale,
                n_ddim_steps=ddim_steps or None,  # 0 -> full DDPM
            )  # Batch x n_samples x (t_ahead + 1) x window x data_dims

            # includes reconstruction of the input as the first step
            if batch_idx == 0:
                data_o = {k: [v.cpu().detach()] for k, v in data_batch.items()}
            else:
                for k in data_o:
                    data_o[k] += [data_batch[k].cpu().detach()]

    data_o = {k: torch.cat(v, dim=0) for k, v in data_o.items()}

    return data_o

# RESULTS_PATH = "/mnt/home/jwu10/working/ceph/results/forecasting/"
# data_o = generate_samples(
#         model_path="{}/pred_ae/1/".format(RESULTS_PATH),
#         epoch=1000,
#         train_val_test="val",
#         n_time_steps=10,
#         n_samples=1,
#         uncond_scale=50,
#         skip_steps=500,
#     )

# name = "ovae_diff/aft_1/"
# config = read.config(RESULTS_PATH + name + "model_config.yaml")
# connectivity = read.config(config["data"]["data_path"] + "mouse_skeleton.yaml")

# skel_offset = torch.tensor(connectivity["OFFSET"], device="cuda", dtype=torch.float32)

# config["model"]["load_model"] = config["out_path"]
# config["model"]["start_epoch"] = 1200
# config["data"]["dataset"] = "parkinsons"
# config["data"]["batch_size"] = 1024
# train_val_test = "val"
# n_time_steps = 1
# loader_dict, model = get.get_data_and_model(
#     config,
#     train_val_test=[train_val_test],
#     data_keys=["x6d", "root", "offsets", "pd_label", "ids"],
#     shuffle=[False],
#     use_default_val_keys=False,
#     n_time_steps=n_time_steps,
# )
# loader = loader_dict[train_val_test]
# loader.dataset.paired_inds = loader.dataset.paired_inds

# window = config["model"]["window"]
# n_keypts = loader.dataset.n_keypts
# kinematic_tree = loader.dataset.kinematic_tree

# v_config = read.config(
#     "/mnt/home/jwu10/working/ceph/results/wu_iclr25/pd/vae/model_config.yaml"
# )
# v_model = scrubvae.get.model(
#     model_config=v_config["model"],
#     load_model=v_config["out_path"],
#     epoch=396,
#     disentangle_config=v_config["disentangle"],
#     n_keypts=loader.dataset.n_keypts,
#     direction_process=v_config["data"]["direction_process"],
#     loss_config=v_config["loss"],
#     arena_size=loader.dataset.arena_size,
#     kinematic_tree=loader.dataset.kinematic_tree,
#     bound=v_config["data"]["normalize"] == "bounded",
#     discrete_classes=loader.dataset.discrete_classes,
#     device="cuda",
#     verbose=-1,
# )

# model.eval()
# v_model.eval()

# n_samples = 1
# downsample = 5
# skip_steps = 0
# v_past, v_future = [], []
# fut_cov, fut_mean = [], []
# with torch.no_grad():
#     uncond_scale = 1
#     for batch_idx, data in enumerate(tqdm.tqdm(loader)):
#         data = {
#             k: v[::downsample].to("cuda")
#             for k, v in data.items()
#             if k not in ["ids", "raw_pose", "target_pose"]
#         }
#         batch_size = data["x6d"].shape[0]
#         # v_past += [
#         #     v_model.encode({k: data[k][:, 0] for k in ["x6d", "root"]})["mu"]
#         #     .cpu()
#         #     .detach()
#         # ]#         # v_future += [
#         #     v_model.encode({k: data[k][:, 1] for k in ["x6d", "root"]})["mu"]
#         #     .cpu()
#         #     .detach()
#         # ]

#         data_o = model.get_samples(
#             data,
#             n_steps_ahead=n_time_steps,
#             n_samples=1,
#             uncond_scale=uncond_scale,
#             skip_steps=skip_steps,
#         )
#         data_o = {k: data_o[k][:, 0, 1] for k in data_o}

#         # print("diff_out {}".format(torch.isnan(data_o["x6d"]).sum()))
#         data_o["root"][..., [0, 1]] -= data_o["root"][..., [0, 1]]

#         pose_o = fwd_kin_cont6d_torch(
#             data_o["x6d"].reshape((-1, n_keypts, 6)),
#             kinematic_tree,
#             data["offsets"][:, 1].reshape((-1, n_keypts, 3)),
#             root_pos=data_o["root"].reshape(-1, 3),
#             do_root_R=True,
#         ).reshape(batch_size, window, n_keypts, 3)

#         forward = pose_o[:, window // 2, 1, :] - pose_o[:, window // 2, 0, :]
#         forward /= torch.norm(forward, dim=-1, keepdim=True)
#         yaw = -torch.arctan2(forward[:, 1], forward[:, 0])

#         local_qtn = inv_kin_torch(
#             pose_o.reshape((-1,) + pose_o.shape[-2:]),
#             connectivity["KINEMATIC_TREE"],
#             skel_offset,
#             forward_indices=[1, 0],
#         ).reshape(pose_o.shape[:-1] + (-1,))

#         fwd_qtn = torch.zeros((len(yaw), 4), dtype=pose_o.dtype, device=pose_o.device)
#         angle2D = torch.cat(
#             [torch.sin(yaw / 2)[:, None], torch.cos(yaw / 2)[:, None]], axis=-1
#         )
#         # angle2D = angle2D.reshape(yaw.shape[:-1] + (-1,))
#         fwd_qtn[:, [-1, 0]] = angle2D
#         fwd_qtn = torch.repeat_interleave(fwd_qtn[:, None, :], window, dim=1)
#         local_qtn[..., 0, :] = qtn.qmul(fwd_qtn, local_qtn[..., 0, :])

#         if torch.isnan(local_qtn).sum() > 0:
#             import pdb

#             pdb.set_trace()

#         data_o["x6d"] = qtn.quaternion_to_cont6d(local_qtn)
#         data_o["root"] = qtn.qrot(fwd_qtn, data_o["root"])

#         v_fut_samples = v_model.encode({k: data_o[k] for k in ["x6d", "root"]})[
#             "mu"
#         ].reshape(batch_size, n_samples, -1)

#         mean = v_fut_samples.mean(dim=1)

#         # data_test = v_model.decode(mean, None)
#         # pose_test = fwd_kin_cont6d_torch(
#         #     data_test["x6d"].reshape((-1, n_keypts, 6)),
#         #     kinematic_tree,
#         #     data["offsets"][:, 1].reshape((-1, n_keypts, 3)),
#         #     root_pos=data_test["root"].reshape(-1, 3),
#         #     do_root_R=True,
#         # ).reshape(batch_size, window, n_keypts, 3)

#         # v_centered = v_fut_samples - mean[:, None, :]
#         # cov = torch.einsum("ijk,ijl->ikl", v_centered, v_centered) / (
#         #     v_centered.shape[1] - 1
#         # )
#         fut_mean += [mean.cpu().detach()]
#         # fut_cov += [cov.detach()]

# # v_past = torch.cat(v_past, dim=0).numpy()
# # np.save(v_config["out_path"] + "/latents/v_past_val.npy", v_past)
# # v_future = torch.cat(v_future, dim=0).numpy()
# # np.save(v_config["out_path"] + "/latents/v_future_val.npy", v_future)
# fut_mean = torch.cat(fut_mean, dim=0).numpy()
# # # fut_cov = torch.cat(fut_cov, dim=0).cpu().detach().numpy()
# np.save(
#     config["out_path"]
#     + "/latents/fut_mean_val_{}_{}.npy".format(
#         config["model"]["start_epoch"], uncond_scale
#     ),
#     fut_mean,
# )
# # np.save(config["out_path"] + "/latents/fut_cov_val.npy", fut_cov)
