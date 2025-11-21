import torch
from scrubvae.train.losses import mpjpe_loss, time_consistency_loss, prior_loss
from scrubvae.data.dataset import fwd_kin_cont6d_torch
import torch.nn.functional as F

from neurobe4.model.model import (
    # ConditionalDiffusion,
    DenoiseDiffusion,
    # NCConditionalDiffusion,
    DiffVAE,
)

from scrubvae.model.residual import ConditionalForecaster, ForecastingVAE

def diffusion_time_consistency_loss(x, x_hat, kinematic_tree, offsets):
    root_hat = torch.zeros_like(offsets[..., 0, 0, :])
    pose = fwd_kin_cont6d_torch(
        x[:, -1],
        kinematic_tree,
        offsets[:, 0],
        root_pos=root_hat,
        do_root_R=False,
        eps=1e-8,
    )

    pose_hat = fwd_kin_cont6d_torch(
        x_hat[:, 0],
        kinematic_tree,
        offsets[:, 0],
        root_pos=root_hat,
        do_root_R=False,
        eps=1e-8,
    )

    loss = torch.sum((pose - pose_hat) ** 2)
    loss = loss / (pose.shape[0] * pose.shape[-1] * pose.shape[-2])

    return loss


def get_batch_loss(model, data, data_o, loss_scale, ddpm):
    is_forecast = data["x6d"].shape[1] == 2
    batch_size = data["x6d"].shape[0]
    batch_loss = {}
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
            data["target_pose"],
            data_o["x6d"],
            model.kinematic_tree,
            data["offsets"][:, 1] if is_forecast else data["offsets"],
        )

    if "diffusion" in loss_scale.keys():
        cond = []
        for k in model.diffusion_cond_keys:
            cond_var = (
                F.one_hot(data[k].ravel().long(), len(model.discrete_classes[k]))
                if k in model.discrete_classes.keys()
                else data[k]
            )
            # Classifier free guidance
            if model.cfg_prob is not None:
                if model.cfg_prob > 0:
                    mask = (
                        torch.rand(batch_size, device=cond_var.device) < model.cfg_prob
                    )
                    cond_var[mask, ...] = 0
            cond += [cond_var]

        cond = torch.cat([data_o["z_past"]] + cond, dim=-1)

        if isinstance(model, DiffVAE):
            cond = cond.detach()

        if "diffusion_time_consistency_a" in loss_scale.keys():
            batch_loss["diffusion"], mean_theta = ddpm.loss(
                eps_model=model.unet, x0=data_o["mu"], cond=cond, return_mean=True
            )

            data_theta = model.decode(torch.squeeze(mean_theta))

            batch_loss["diffusion_time_consistency_a"] = (
                diffusion_time_consistency_loss(
                    data["x6d"][:, 0],
                    data_theta["x6d"],
                    model.kinematic_tree,
                    data["offsets"][:, 1],
                )
            )
        else:
            batch_loss["diffusion"] = ddpm.loss(
                eps_model=model.unet, x0=data_o["mu"], cond=cond, return_mean=False
            )

        if "diffusion_time_consistency_b" in loss_scale.keys():
            mean_theta = ddpm.t0_time_consistency_loss(
                eps_model=model.unet, x0=data_o["mu"], cond=cond
            )
            data_theta = model.decode(torch.squeeze(mean_theta))
            batch_loss["diffusion_time_consistency_b"] = (
                diffusion_time_consistency_loss(
                    data["x6d"][:, 0],
                    data_theta["x6d"],
                    model.kinematic_tree,
                    data["offsets"][:, 1],
                )
            )

    if "time_consistency" in loss_scale.keys():
        batch_loss["time_consistency"] = time_consistency_loss(
            data["x6d"][:, 1],
            data_o["x6d"],
            model.kinematic_tree,
            data["offsets"][:, 1] if is_forecast else data["offsets"],
        )

    if "root" in loss_scale.keys():
        batch_loss["root"] = (
            torch.nn.MSELoss(reduction="sum")(
                data_o["root"], data["root"][:, 1] if is_forecast else data["root"]
            )
            / batch_size
        )

    batch_loss["total"] = sum(
        [loss_scale[k] * batch_loss[k] for k in batch_loss.keys() if loss_scale[k] != 0]
    )

    return batch_loss
