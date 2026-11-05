import torch
from scrubvae.train.losses import mpjpe_loss, time_consistency_loss, prior_loss
from scrubvae.data.dataset import fwd_kin_cont6d_torch
import torch.nn.functional as F

from neurobe4.model.model import (
    DenoiseDiffusion,
    DiffVAE,
)

from neurobe4.model.model import ConditionalForecaster, ForecastingVAE, ConditionalDiffusion

def diffusion_time_consistency_loss(x, x_hat, kinematic_tree, offsets):
    root_hat = torch.zeros_like(offsets[..., 0, 0, :])
    pose = fwd_kin_cont6d_torch(
        x[:, -1],
        kinematic_tree,
        offsets,
        root_pos=root_hat,
        do_root_R=False,
        eps=1e-8,
    )

    pose_hat = fwd_kin_cont6d_torch(
        x_hat[:, 0],
        kinematic_tree,
        offsets,
        root_pos=root_hat,
        do_root_R=False,
        eps=1e-8,
    )

    loss = torch.sum((pose - pose_hat) ** 2)
    loss = loss / (pose.shape[0] * pose.shape[-1] * pose.shape[-2])

    return loss


def get_batch_loss(model, data, data_o, loss_scale):
    is_forecast = getattr(model, "is_forecasting", False)
    batch_size = data[list(data.keys())[0]].shape[0]
    batch_loss = {}
    data_gt = data

    if "prior" in loss_scale.keys():
        if "L" in data_o.keys():
            batch_loss["prior"] = prior_loss(data_o["mu"], data_o["L"])
        elif "beta_dist" in data_o.keys():
            p = torch.distributions.Beta(
                torch.ones_like(data_o["alpha"]),
                torch.ones_like(data_o["beta"]),
            )
            batch_loss["prior"] = (
                torch.distributions.kl_divergence(data_o["beta_dist"], p).sum(-1).sum()
                / batch_size
            )

    if "jpe" in loss_scale.keys():
        batch_loss["jpe"] = mpjpe_loss(
            data_gt["target_pose"],
            data_o["x6d"],
            model.kinematic_tree,
            data_gt["offsets"],
        )

    if "diffusion" in loss_scale.keys():
        # Forecast horizon (in windows) of each sample's target: drawn per
        # sample with random_tau, else None (target is window 1)
        tau = data_gt.get("tau")

        # if isinstance(model, (ConditionalDiffusion, DiffVAE)):
        # Past, extra conditioning variables (read from data_gt) and tau, with
        # their classifier-free guidance dropout
        cond = model.build_cond(data_o["z_past"], data_gt, tau)

        # Normalization rows are indexed by horizon: without random_tau every
        # target is window 1 (row 0 of a per-tau table is the identity)
        if tau is None:
            tau = torch.ones(batch_size, dtype=torch.long, device=data_o["mu"].device)
        # else:
        #     cond = torch.cat([data_o["z_past"]] + cond, dim=-1)

        if isinstance(model, DiffVAE) and model.forecasting_type == "velocity":
            batch_loss["diffusion"] = model.ddpm.loss(
                model=model.unet, x0=data_o["mu"] - data_o["z_past"], cond=cond, tau=tau,
            )
        else:
            batch_loss["diffusion"] = model.ddpm.loss(
                model=model.unet, x0=data_o["mu"], cond=cond, tau=tau,
            )

    if "root" in loss_scale.keys():
        batch_loss["root"] = (
            torch.nn.MSELoss(reduction="sum")(
                data_o["root"].squeeze(), data_gt["root"][:, 1].squeeze() if is_forecast else data_gt["root"].squeeze()
            )
            / batch_size
        )

    batch_loss["total"] = sum(
        [loss_scale[k] * batch_loss[k] for k in batch_loss.keys() if loss_scale[k] != 0]
    )

    return batch_loss
