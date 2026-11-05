from scrubvae.model.residual import (
    VAE,
    ResidualDecoder,
    ResVAE,
    CholeskyL,
)
from neurobe4.model.residual import ResidualEncoder

import math
from typing import Tuple, Optional, Union, Dict
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from neurobe4.model.unet_sd import UNetModel, LatentUNet
from neurobe4.data.process import process_batch_output

# from labml_nn.diffusion.ddpm.utils import gather

def build_denoiser(z_dim: int, d_cond: int, expand_latent=None) -> nn.Module:
    """
    Denoiser for z_dim latent vectors. expand_latent=None: the original U-Net,
    which treats the latent as a 1-channel sequence of length z_dim
    (attention at levels 0 and 1). expand_latent=[channels, length]: LatentUNet,
    which first maps the latent to a learned (channels, length) feature map
    and uses attention at every level.
    """
    if expand_latent is None:
        return UNetModel(
            in_channels=1,
            out_channels=1,
            channels=z_dim,
            attention_levels=[0, 1],
            n_res_blocks=1,
            channel_multipliers=[1, 2, 2],
            n_heads=4,
            tf_layers=1,
            d_cond=d_cond,
        )
    grid_channels, grid_length = expand_latent
    return LatentUNet(
        latent_dim=z_dim,
        grid_channels=grid_channels,
        grid_length=grid_length,
        n_res_blocks=1,
        channel_multipliers=[1, 2, 2],
        n_heads=4,
        tf_layers=1,
        d_cond=d_cond,
    )


def build_cond_vars(model, data, batch_size: int, device) -> list:
    """
    Extra diffusion conditioning variables (model.diffusion_cond_keys) for a
    denoiser's build_cond: [(B, diffusion_cond_dim)] -- discrete keys one-hot,
    others as-is -- or [] if there are none. data=None gives the null
    (all-zero) variables. During training, each key is zeroed for an
    independent random cfg_prob fraction of samples (classifier-free guidance).
    """
    keys = model.diffusion_cond_keys or []
    if len(keys) == 0:
        return []
    if data is None:
        return [torch.zeros(batch_size, model.diffusion_cond_dim, device=device)]
    discrete = model.discrete_classes or {}
    cond_vars = []
    for k in keys:
        if k in discrete:
            v = F.one_hot(data[k].ravel().long(), len(discrete[k]))
        else:
            v = data[k]
        v = v.reshape(batch_size, -1).float()
        if model.training and model.cfg_prob is not None and model.cfg_prob > 0:
            mask = torch.rand(batch_size, device=v.device) < model.cfg_prob
            v = torch.where(mask[:, None], torch.zeros_like(v), v)
        cond_vars.append(v)
    return [torch.cat(cond_vars, dim=-1)]


# Conditioning layers renamed after some checkpoints were saved: old -> new
LEGACY_COND_NAMES = {
    "past_token_embed": "past_embed",
    "past_token_pos": "past_pos",
    "cond_token_proj": "cond_proj",
}


def rename_legacy_cond_keys(state_dict, prefix: str):
    """Rename a checkpoint's legacy conditioning-layer keys (in place) so they
    load into the current layer names"""
    for old, new in LEGACY_COND_NAMES.items():
        for key in [k for k in state_dict if k.startswith(prefix + old)]:
            new_key = prefix + new + key[len(prefix + old):]
            if new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(key)


class TauEmbedding(nn.Module):
    """Forecast horizon tau (B,) -> conditioning token (B, dim): sinusoidal
    embedding followed by a small MLP"""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=tau.device)
            / half
        )
        args = tau.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


class DenoiseDiffusion(nn.Module):
    """
    ## Denoise Diffusion
    """

    def __init__(
        self,
        n_steps: int,
        latent_dim: int = 64,
        latent_norm: Optional[Dict] = None,
        diffusion_type="eps_pred",
    ):
        """
        * `model` is $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        * `n_steps` is $t$
        * `device` is the device to place constants on
        """
        super().__init__()
        self.diffusion_type = diffusion_type

        # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
        self.register_buffer("beta", torch.linspace(0.0001, 0.02, n_steps))

        # $\alpha_t = 1 - \beta_t$
        self.register_buffer("alpha", 1.0 - self.beta)
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))
        # $T$
        self.n_steps = n_steps
        # $\sigma^2 = \beta$
        self.register_buffer("sigma2", self.beta)

        # Latent normalization statistics: (1, latent_dim), or one row per
        # forecast horizon tau, (n_tau, latent_dim), indexed by tau
        if latent_norm is not None:
            if "scalar" in latent_norm.keys():
                scalar = torch.as_tensor(latent_norm["scalar"], dtype=torch.float32)
                self.register_buffer("latent_mu", torch.zeros(latent_dim, requires_grad=False)[None,:])
                self.register_buffer("latent_std", torch.ones(latent_dim, requires_grad=False)[None, :] * scalar.reshape(-1, 1))
            elif "mu" in latent_norm.keys() and "std" in latent_norm.keys():
                self.register_buffer("latent_mu", latent_norm["mu"].reshape(-1, latent_dim))
                self.register_buffer("latent_std", latent_norm["std"].reshape(-1, latent_dim))
        else:
            self.register_buffer("latent_mu", torch.zeros(latent_dim, requires_grad=False)[None,:])
            self.register_buffer("latent_std", torch.ones(latent_dim, requires_grad=False)[None,:])

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Normalization statistics may be per-tau tables of a different number
        # of rows than this instance was built with (e.g. when evaluating
        # without recomputing them) -- adopt the checkpoint's shape
        for name in ("latent_mu", "latent_std"):
            key = prefix + name
            if key in state_dict and state_dict[key].shape != getattr(self, name).shape:
                setattr(self, name, torch.empty_like(state_dict[key]))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def norm_stats(self, tau: Optional[torch.Tensor] = None):
        """Normalization (mu, std) for each sample: (B, latent_dim) rows by tau
        for per-tau statistics, else the shared (1, latent_dim) row"""
        if self.latent_mu.shape[0] == 1 or tau is None:
            return self.latent_mu[:1], self.latent_std[:1]
        return self.latent_mu[tau], self.latent_std[tau]

    def normalize(self, x0: torch.Tensor, tau: Optional[torch.Tensor] = None):
        mu, std = self.norm_stats(tau)
        return (x0 - mu) / std

    def denormalize(self, x: torch.Tensor, tau: Optional[torch.Tensor] = None):
        mu, std = self.norm_stats(tau)
        return x * std + mu

    def q_xt_x0(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        #### Get $q(x_t|x_0)$ distribution

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alpha_bar, t)
        #
        return mean, var

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None
    ):
        """
        #### Sample from $q(x_t|x_0)$

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if eps is None:
            eps = torch.randn_like(x0)

        # get $q(x_t|x_0)$
        mean, var = self.q_xt_x0(x0, t)
        # Sample from $q(x_t|x_0)$
        return mean + (var**0.5) * eps

    def _to_eps(self, out: torch.Tensor, xt: torch.Tensor, t: torch.Tensor):
        """Denoiser output -> predicted noise (converting from v for v_pred)"""
        if self.diffusion_type == "v_pred":
            a_t = gather(self.alpha_bar, t) ** 0.5
            var = 1 - gather(self.alpha_bar, t)
            return out * a_t + xt[:, None, :] * (var**0.5)
        return out

    def get_eps(
        self,
        model,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor = None,
        uncond_scale: float = 1,
        uncond_tau: torch.Tensor = None,
        tau_scale: float = 1,
    ):
        """
        Classifier-free guidance with up to two scales. cond = c(past, tau),
        uncond = c(null past, tau), uncond_tau = c(null past, null tau):

            eps = e(null, null) + tau_scale * [e(null, tau) - e(null, null)]
                                + uncond_scale * [e(past, tau) - e(null, tau)]

        Passes whose scale is 1 are skipped: tau_scale = 1 reduces to ordinary
        past guidance against uncond, and both scales = 1 to a single
        conditional pass.

        A list cond = [c_0, ..., c_k] (nested conditioning sets, e.g. from
        SocialConditionalDiffusion) with uncond_scale = [s_1, ..., s_k]
        instead uses chained guidance (see get_eps_chain).
        """
        if isinstance(cond, (list, tuple)):
            return self.get_eps_chain(model, xt, t, cond, uncond_scale)
        guide_past = uncond is not None and uncond_scale != 1.0
        guide_tau = uncond_tau is not None and tau_scale != 1.0
        if not (guide_past or guide_tau):
            e_t = self._to_eps(model(xt, t, cond), xt, t)
            if torch.any(torch.isnan(e_t)):
                import pdb

                pdb.set_trace()
            return e_t

        if not guide_tau:
            # Duplicate $x_t$ and $t$
            x_in = torch.cat([xt] * 2, dim=0)
            t_in = torch.cat([t] * 2, dim=0)
            # Concatenated $c$ and $c_u$
            c_in = torch.cat([uncond, cond], dim=0)
            # Get $\epsilon_\text{cond}(x_t, c)$ and $\epsilon_\text{cond}(x_t, c_u)$
            out_uncond, out_cond = model(x_in, t_in, c_in).chunk(2)
            e_t_uncond = self._to_eps(out_uncond, xt, t)
            e_t_cond = self._to_eps(out_cond, xt, t)
            # Calculate
            # $$\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$$
            return e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)

        # Two-scale guidance: e(null, null), e(null, tau), e(past, tau)
        x_in = torch.cat([xt] * 3, dim=0)
        t_in = torch.cat([t] * 3, dim=0)
        c_in = torch.cat([uncond_tau, uncond, cond], dim=0)
        out_nn, out_nt, out_pt = model(x_in, t_in, c_in).chunk(3)
        e_nn = self._to_eps(out_nn, xt, t)
        e_nt = self._to_eps(out_nt, xt, t)
        e_pt = self._to_eps(out_pt, xt, t)
        return e_nn + tau_scale * (e_nt - e_nn) + uncond_scale * (e_pt - e_nt)

    def get_eps_chain(self, model, xt: torch.Tensor, t: torch.Tensor, conds, scales):
        """
        Chained classifier-free guidance over nested conditioning sets
        c_0 (fewest conditions) ... c_k (all), each (B, n_tokens, d_cond):

            eps = e(c_0) + sum_i scales[i - 1] * [e(c_i) - e(c_{i-1})]

        so each scale weighs what its condition adds beyond the previous
        ones (the two-scale guidance of get_eps is the chain [null, tau,
        past + tau]). Leading scales of 1 cancel their base pass, so e.g.
        all scales 1 is a single conditional pass.
        """
        conds, scales = list(conds), [float(s) for s in scales]
        assert len(scales) == len(conds) - 1, "one scale per condition added"
        while len(scales) > 0 and scales[0] == 1.0:
            conds.pop(0)
            scales.pop(0)
        k = len(conds)
        out = model(torch.cat([xt] * k, dim=0), torch.cat([t] * k, dim=0), torch.cat(conds, dim=0))
        e = [self._to_eps(o, xt, t) for o in out.chunk(k)]
        eps = e[0]
        for i, s in enumerate(scales, start=1):
            eps = eps + s * (e[i] - e[i - 1])
        return eps

    def p_sample(
        self,
        model,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor = None,
        uncond_scale: float = 1,
        uncond_tau: torch.Tensor = None,
        tau_scale: float = 1,
    ):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        \begin{align}
        \textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
        \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
        \textcolor{lightgreen}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """
        xt = xt[:, None, :]
        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        eps_theta = self.get_eps(
            model, xt.squeeze(), t, cond, uncond, uncond_scale, uncond_tau, tau_scale
        )
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t)
        # $\alpha_t$
        alpha = gather(self.alpha, t)
        # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5
        # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
        #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
        mean = 1 / (alpha**0.5) * (xt - eps_coef * eps_theta)
        # $\sigma^2$
        var = gather(self.sigma2, t)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        eps = torch.randn(xt.shape, device=xt.device)
        # Sample; no noise at the final step (t = 0), whose output is the sample
        nonzero_t = (t > 0).to(xt.dtype).reshape(-1, 1, 1)
        sample = mean + nonzero_t * (var**0.5) * eps

        return sample

    def ddim_timesteps(
        self, n_ddim_steps: int, spacing: str = "quadratic", device=None
    ) -> torch.Tensor:
        """
        Descending timesteps for DDIM sampling: n_ddim_steps steps from
        n_steps - 1 down to 0 (n_ddim_steps = n_steps gives every step). Pair
        each with the next one as t_prev (-1 after the last).

        spacing : "uniform" (evenly spaced) or "quadratic" (denser near t = 0;
            with this linear beta schedule it reduces the few-step
            under-dispersion of DDIM samples)
        """
        if spacing == "uniform":
            steps = torch.linspace(0, self.n_steps - 1, n_ddim_steps, device=device)
        elif spacing == "quadratic":
            steps = (
                torch.linspace(0, (self.n_steps - 1) ** 0.5, n_ddim_steps, device=device)
                ** 2
            )
        else:
            raise ValueError(f"Unknown spacing {spacing}")
        return steps.round().long().unique().flip(0)

    def ddim_sample(
        self,
        model,
        xt: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor = None,
        uncond_scale: float = 1,
        uncond_tau: torch.Tensor = None,
        tau_scale: float = 1,
        eta: float = 0.0,
    ):
        """
        #### DDIM step from timestep t to t_prev < t (Song et al., 2021)

        Same model and guidance arguments as p_sample, but jumps directly to
        an earlier timestep t_prev (t_prev = -1 means the final, noise-free
        sample), so sampling can use a short subsequence of timesteps (see
        ddim_timesteps):

            x0_hat = (x_t - sqrt(1 - abar_t) eps) / sqrt(abar_t)
            x_prev = sqrt(abar_prev) x0_hat
                     + sqrt(1 - abar_prev - sigma^2) eps + sigma z
            sigma  = eta * sqrt((1 - abar_prev) / (1 - abar_t))
                         * sqrt(1 - abar_t / abar_prev)

        eta = 0 is deterministic DDIM; eta = 1 with every timestep recovers
        DDPM-like stochastic sampling.

        xt : (B, latent_dim); returns (B, 1, latent_dim), like p_sample.
        """
        xt = xt[:, None, :]
        eps_theta = self.get_eps(
            model, xt.squeeze(1), t, cond, uncond, uncond_scale, uncond_tau, tau_scale
        )
        alpha_bar = gather(self.alpha_bar, t)
        # abar at t_prev; 1 (no noise) past the start of the chain
        alpha_bar_prev = torch.where(
            t_prev.reshape(-1, 1, 1) >= 0,
            gather(self.alpha_bar, t_prev.clamp(min=0)),
            torch.ones_like(alpha_bar),
        )

        x0_hat = (xt - (1 - alpha_bar) ** 0.5 * eps_theta) / alpha_bar**0.5
        sigma = (
            eta
            * ((1 - alpha_bar_prev) / (1 - alpha_bar)) ** 0.5
            * (1 - alpha_bar / alpha_bar_prev) ** 0.5
        )
        dir_xt = (1 - alpha_bar_prev - sigma**2).clamp(min=0) ** 0.5 * eps_theta
        noise = torch.randn(xt.shape, device=xt.device)
        return alpha_bar_prev**0.5 * x0_hat + dir_xt + sigma * noise

    def sample(
        self,
        model,
        xt: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor = None,
        uncond_scale: float = 1,
        uncond_tau: torch.Tensor = None,
        tau_scale: float = 1,
        # skip_steps: int = 0,
        n_ddim_steps: Optional[int] = None,
        ddim_eta: float = 0.0,
        ddim_spacing: str = "quadratic",
    ) -> torch.Tensor:
        """
        Full reverse process from noise xt (B, latent_dim) to a (normalized)
        sample (B, latent_dim), with the guidance arguments of p_sample.

        n_ddim_steps=None: DDPM (p_sample) over every timestep, starting
            skip_steps below the last.
        n_ddim_steps=k: DDIM (ddim_sample) over k timesteps (see
            ddim_timesteps(k, ddim_spacing)) with noise level ddim_eta;
            skip_steps is ignored.
        """
        B = xt.shape[0]
        guidance = dict(
            uncond=uncond,
            uncond_scale=uncond_scale,
            uncond_tau=uncond_tau,
            tau_scale=tau_scale,
        )
        if n_ddim_steps is None:
            time_steps = torch.arange(self.n_steps, device=xt.device).flip(0)#[skip_steps:]
            for step in time_steps:
                t = xt.new_full((B,), int(step), dtype=torch.long)
                xt = self.p_sample(model, xt, t, cond, **guidance).squeeze(1)
            return xt

        time_steps = self.ddim_timesteps(n_ddim_steps, ddim_spacing, device=xt.device)
        for i, step in enumerate(time_steps):
            t_prev = int(time_steps[i + 1]) if i + 1 < len(time_steps) else -1
            t = xt.new_full((B,), int(step), dtype=torch.long)
            tp = xt.new_full((B,), t_prev, dtype=torch.long)
            xt = self.ddim_sample(model, xt, t, tp, cond, eta=ddim_eta, **guidance).squeeze(1)
        return xt

    def loss(
        self,
        model,
        x0: torch.Tensor,
        cond: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
    ):
        """
        tau : (B,) forecast horizons, selecting per-tau normalization rows
        #### Simplified Loss

        $$L_{\text{simple}}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """
        # Get batch size
        # Latent normalization (identity unless latent_norm was given; per-tau
        # rows if given per horizon); inverted after sampling in each model's
        # get_samples
        x0 = self.normalize(x0, tau)

        x0 = x0[:, None, :]
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        if self.diffusion_type == "v_pred":
            mean, var = self.q_xt_x0(x0, t)
            target = (gather(self.alpha_bar, t) ** 0.5) * noise - (var**0.5) * x0
        else:
            # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
            target = noise
        pred = model(
            xt.squeeze(), t, cond
        )  # self.get_eps(model, xt.squeeze(), t, cond, uncond, uncond_scale)
        
        return F.mse_loss(target, pred)

    def forward(self):
        return None


class ResidualBlockMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        activation="prelu",
    ):

        super(ResidualBlockMLP, self).__init__()
        self.residual = nn.Sequential(
            nn.Linear(
                in_dim,
                out_dim // 2,
                bias=True,
            ),
            nn.BatchNorm1d(out_dim // 2, eps=1e-4),
            nn.Tanh() if activation == "tanh" else nn.PReLU(),
            nn.Linear(
                out_dim // 2,
                out_dim,
                bias=True,
            ),
        )

        self.skip = nn.Linear(
            in_dim,
            out_dim,
            bias=True,
        )

        self.add = nn.Sequential(
            nn.BatchNorm1d(out_dim, eps=1e-4),
            nn.Tanh() if activation == "tanh" else nn.PReLU(),
        )

    def forward(self, x):
        skip = self.skip(x)
        x = self.residual(x)
        return self.add(x + skip)


class ResidualBlockTransposeMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        activation="prelu",
    ):
        super(ResidualBlockTransposeMLP, self).__init__()

        self.residual = nn.Sequential(
            nn.Linear(
                in_dim,
                in_dim // 2,
                bias=True,
            ),
            nn.BatchNorm1d(in_dim // 2, eps=1e-4),
            nn.Tanh() if activation == "tanh" else nn.PReLU(),
            nn.Linear(
                in_dim // 2,
                out_dim,
                bias=True,
            ),
        )

        self.skip = nn.Linear(
            in_dim,
            out_dim,
            bias=True,
        )

        self.add = nn.Sequential(
            nn.BatchNorm1d(out_dim, eps=1e-4),
            nn.Tanh() if activation == "tanh" else nn.PReLU(),
        )

    def forward(self, x):
        skip = self.skip(x)
        x = self.residual(x)
        return self.add(x + skip)


class ResidualEncoderMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        layer_dims=[64, 128, 256, 512, 1024],
        z_dim=128,
        activation="prelu",
        is_diag=False,
        prior="gaussian",
    ):
        super(ResidualEncoderMLP, self).__init__()
        self.prior = prior
        self.conv_in = nn.Linear(in_dim, layer_dims[0])
        self.activation = nn.Tanh() if activation == "tanh" else nn.PReLU()

        layers = []
        for i in range(len(layer_dims) - 1):
            layers += [
                ResidualBlockMLP(
                    layer_dims[i],
                    layer_dims[i + 1],
                    activation,
                )
            ]
        self.res_layers = nn.Sequential(*layers)

        if prior == "gaussian":
            sig_dim = z_dim if is_diag else z_dim * (z_dim + 1) // 2
            self.fc_mu = nn.Linear(layer_dims[-1], z_dim)
            self.fc_sigma = nn.Sequential(
                nn.Linear(layer_dims[-1], sig_dim), CholeskyL(z_dim, is_diag)
            )
        elif prior == "beta":
            self.fc_alpha = nn.Linear(layer_dims[-1], z_dim)
            self.fc_beta = nn.Linear(layer_dims[-1], z_dim)
        elif prior is None:
            self.fc = nn.Linear(layer_dims[-1], z_dim)

    def forward(self, x):
        x = self.activation(self.conv_in(x))
        x = self.res_layers(x)
        if self.prior == "gaussian":
            mu = self.fc_mu(x)
            sigma = self.fc_sigma(x)
            return mu, sigma
        elif self.prior == "beta":
            # Making sure beta distribution has one mode
            alpha = F.softplus(self.fc_alpha(x)) + 1
            beta = F.softplus(self.fc_beta(x)) + 1
            return alpha, beta
        elif self.prior is None:
            return self.fc(x)
        return 0


class ResidualDecoderMLP(nn.Module):
    def __init__(
        self,
        out_dim,
        layer_dims=[64, 128, 256, 512, 1024],
        z_dim=64,
        activation="prelu",
        conditional_dim=0,
    ):
        super(ResidualDecoderMLP, self).__init__()
        self.conditional_dim = conditional_dim

        self.fc_in = nn.Linear(z_dim + conditional_dim, layer_dims[-1])

        layers = []
        for i in range(1, len(layer_dims)):
            layers += [
                ResidualBlockTransposeMLP(
                    layer_dims[-i],
                    layer_dims[-i - 1],
                    activation=activation,
                )
            ]
        self.res_layers = nn.Sequential(*layers)
        self.conv_out = nn.Linear(layer_dims[0], out_dim)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.res_layers(x)
        x = torch.tanh(self.conv_out(x))
        return x


class MLPVAE(VAE):
    def __init__(
        self,
        in_dim,
        layer_dims=[128, 256, 512, 1024],
        z_dim=64,
        activation="prelu",
        is_diag=False,
        prior="gaussian",
        arena_size=None,
        kinematic_tree=None,
    ):
        super().__init__(prior=prior)
        self.in_dim = in_dim
        self.layer_dims = layer_dims
        self.z_dim = z_dim
        self.activation = activation
        self.register_buffer("arena_size", arena_size)
        self.kinematic_tree = kinematic_tree

        self.encoder = ResidualEncoderMLP(
            in_dim=in_dim,
            layer_dims=layer_dims,
            z_dim=z_dim,
            activation=activation,
            is_diag=is_diag,
            prior=prior,
        )
        self.decoder = ResidualDecoderMLP(
            out_dim=in_dim,
            layer_dims=layer_dims,
            z_dim=z_dim,
            activation=activation,
            conditional_dim=0,
        )

    def normalize_root(self, root):
        norm_root = root - self.arena_size[0]
        norm_root = 2 * norm_root / (self.arena_size[1] - self.arena_size[0]) - 1
        return norm_root

    def inv_normalize_root(self, norm_root):
        root = 0.5 * (norm_root + 1) * (self.arena_size[1] - self.arena_size[0])
        root += self.arena_size[0]
        return root

    def encode(self, data):
        if self.arena_size is not None:
            norm_root = self.normalize_root(data["root"])

            x_in = torch.cat(
                (data["x6d"].view(data["x6d"].shape[:2] + (-1,)), norm_root), axis=-1
            )
        else:
            x_in = data["x6d"]

        data_o = {}
        data_o[self.dist_params[0]], data_o[self.dist_params[1]] = self.encoder(
            x_in.reshape(-1, x_in.shape[-1])
        )

        if self.prior == "beta":
            # Renormalizing to be between (-1, 1)
            data_o["mu"] = (data_o["alpha"] - 1 + 1e-8) / (
                data_o["alpha"] + data_o["beta"] - 2 + 2e-8
            ) * 2 - 1

        return data_o

    def decode(self, z, data):
        data_o = {}
        x_hat = self.decoder(z).moveaxis(-1, 1)

        if self.arena_size is None:
            x6d = x_hat
        else:
            x6d = x_hat[..., :-3]
            data_o["root"] = self.inv_normalize_root(x_hat[..., -3:]).reshape(
                z.shape[0], 1, 3
            )

        x6d = F.normalize(x6d.reshape(z.shape[0], 1, -1, 2, 3), dim=-1)
        data_o["x6d"] = x6d.reshape(z.shape[0], 1, -1, 6)

        return data_o


class DiffVAE(ResVAE):
    def __init__(
        self,
        in_channels,
        ch=[64, 128, 256, 512, 1024],
        kernel=5,
        z_dim=128,
        window=200,
        activation="prelu",
        is_diag=False,
        conditional_dim=0,
        init_dilation=None,
        disentangle=None,
        kinematic_tree=None,
        arena_size=None,
        disentangle_keys=None,
        conditional_keys=None,
        discrete_classes=None,
        prior="gaussian",
        diffusion_cond_keys=None,
        diffusion_cond_dim=None,
        cfg_prob=0,
        diffusion_type="e_pred",
        forecasting_type="velocity",
        latent_norm=None,
        expand_latent=None,
        condition_tau=False,
        cfg_prob_tau=0.0,
    ):
        """
        expand_latent : [channels, length] or None
            Denoiser layout (see build_denoiser). None keeps the original U-Net.
        condition_tau : bool
            Condition the denoiser on the forecast horizon tau via one more
            conditioning token (see TauEmbedding), replaced by a learned null
            token for a random cfg_prob_tau fraction of training samples
            (independently of the past's cfg_prob), so tau can be guided.
        """

        super().__init__(
            in_channels=in_channels,
            ch=ch,
            kernel=kernel,
            z_dim=z_dim,
            window=window,
            activation=activation,
            is_diag=is_diag,
            conditional_dim=conditional_dim,
            init_dilation=init_dilation,
            disentangle=disentangle,
            kinematic_tree=kinematic_tree,
            arena_size=arena_size,
            disentangle_keys=disentangle_keys,
            conditional_keys=conditional_keys,
            discrete_classes=discrete_classes,
            prior=prior,
        )
        self.is_forecasting = True
        self.forecasting_type = forecasting_type

        self.ddpm = DenoiseDiffusion(
            n_steps=1000,
            latent_dim=z_dim,
            diffusion_type=diffusion_type,
            latent_norm=latent_norm,
        )
        self.diffusion_cond_keys = diffusion_cond_keys
        self.diffusion_cond_dim = diffusion_cond_dim
        self.cfg_prob = cfg_prob
        self.null_token = None

        self.unet = build_denoiser(z_dim, d_cond=z_dim, expand_latent=expand_latent)

        # The past latent is given to the denoiser as z_dim tokens, one per
        # latent dimension: each value is lifted to a vector by a small MLP plus
        # a learned per-dimension positional embedding, so cross-attention can
        # relate each dimension of the noisy future latent to the past latent's
        # dimensions (a single token would reduce attention to a fixed
        # projection). Extra diffusion_cond_keys are projected to one more token.
        self.past_embed = nn.Sequential(
            nn.Linear(1, z_dim), nn.SiLU(), nn.Linear(z_dim, z_dim)
        )
        self.past_pos = nn.Parameter(0.02 * torch.randn(1, z_dim, z_dim))
        if diffusion_cond_dim > 0:
            self.cond_proj = nn.Linear(diffusion_cond_dim, z_dim)

        self.condition_tau = condition_tau
        self.cfg_prob_tau = cfg_prob_tau
        if condition_tau:
            self.tau_embed = TauEmbedding(z_dim)
            self.null_tau_token = nn.Parameter(0.02 * torch.randn(1, z_dim))

    def tau_token(self, tau: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        """(B, 1, z_dim) tau token; the learned null token if tau is None, and
        for a random cfg_prob_tau fraction of samples during training"""
        null = self.null_tau_token.expand(batch_size, -1)
        if tau is None:
            return null[:, None]
        token = self.tau_embed(tau)
        if self.training and self.cfg_prob_tau > 0:
            mask = torch.rand(batch_size, device=token.device) < self.cfg_prob_tau
            token = torch.where(mask[:, None], null, token)
        return token[:, None]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        rename_legacy_cond_keys(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def build_cond(
        self,
        z_past: torch.Tensor,
        data: Optional[Dict] = None,
        tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Denoiser conditioning tokens, (B, z_dim [+ 1] [+ 1], z_dim), from z_past
        (B, z_dim), the extra conditioning variables diffusion_cond_keys read
        from data (None -> null variables; see build_cond_vars) and, if
        condition_tau, the horizon tau (B,) (None -> null tau). During
        training, z_past is replaced by the null past for a random cfg_prob
        fraction of samples (classifier-free guidance) -- only here, so callers
        keep the true z_past (e.g. for the velocity target) -- and tau and each
        extra variable for independent fractions.
        """
        cond_vars = build_cond_vars(self, data, z_past.shape[0], z_past.device)
        # Classifier-free guidance: null past for a random cfg_prob fraction
        if self.training and self.cfg_prob is not None and self.cfg_prob > 0:
            mask = torch.rand(z_past.shape[0], device=z_past.device) < self.cfg_prob
            z_past = torch.where(mask[:, None], self.get_null().to(z_past.dtype), z_past)
        tokens = self.past_embed(z_past[..., None]) + self.past_pos
        if len(cond_vars) > 0:
            cond_token = self.cond_proj(torch.cat(cond_vars, dim=-1))
            tokens = torch.cat([tokens, cond_token[:, None]], dim=1)
        if self.condition_tau:
            tokens = torch.cat([tokens, self.tau_token(tau, z_past.shape[0])], dim=1)
        return tokens

    def get_null(self):
        """z_past of an all-zero (null) past window, (1, z_dim)"""
        device = next(self.encoder.parameters()).device
        null_param1, null_param2 = self.encoder(
            torch.zeros((1, self.in_channels, self.window), device=device)
        )
        if self.prior == "beta":
            return (null_param1 - 1 + 1e-8) / (
                null_param1 + null_param2 - 2 + 2e-8
            ) * 2 - 1
        return null_param1

    def get_samples(
        self,
        data,
        n_future_windows,
        n_samples,
        uncond_scale,
        tau: int = 1,
        tau_scale: float = 1,
        n_ddim_steps: Optional[int] = None,
        ddim_eta: float = 0.0,
        ddim_spacing: str = "quadratic",
    ):
        """
        Autoregressive forecast: each of n_future_windows steps predicts the latent
        tau windows (tau * n_skip frames) after the previous one.
        uncond_scale guides the past; tau_scale guides tau (condition_tau
        models only; 1 = ordinary conditioning on tau).
        n_ddim_steps=None samples each step with DDPM over all timesteps
        (minus skip_steps); n_ddim_steps=k uses DDIM with k timesteps, noise
        level ddim_eta and ddim_spacing (see DenoiseDiffusion.sample).
        """
        batch_size = data["x6d"].shape[0]
        tau_t = torch.full(
            (batch_size * n_samples,), tau, dtype=torch.long, device=data["x6d"].device
        )

        if self.null_token is None:
            self.null_token = self.get_null()

        past_data = {k: v[:, 0] for k, v in data.items() if k in ["x6d", "root"]}
        # Extra conditioning variables, in the same (batch-major, sample-minor)
        # order as the latents below
        cond_data = {
            k: data[k].repeat_interleave(n_samples, dim=0)
            for k in (self.diffusion_cond_keys or [])
        }

        norm_root = self.normalize_root(past_data["root"])
        x_in = torch.cat(
            (past_data["x6d"].view(past_data["x6d"].shape[:-2] + (-1,)), norm_root),
            axis=-1,
        )
        null_past = self.null_token.expand(batch_size * n_samples, -1)
        uncond = self.build_cond(null_past, None, tau_t)
        uncond_tau = (
            self.build_cond(null_past, None, None) if self.condition_tau else None
        )

        init_mu = self.encoder(x_in.moveaxis(-2, -1))[0].reshape(
            batch_size, 1, 1, self.z_dim
        )
        data_o = {
            "mu": [init_mu.repeat(1, n_samples, 1, 1)]
        }  # Batch x n_samples x t_ahead x z_dim
        for step_ahead in range(1, n_future_windows + 1):
            cond = self.build_cond(
                data_o["mu"][-1].reshape(-1, self.z_dim), cond_data, tau_t
            )

            xt = torch.randn_like(data_o["mu"][-1].reshape(-1, self.z_dim))
            xt = self.ddpm.sample(
                self.unet,
                xt,
                cond,
                uncond=uncond,
                uncond_scale=uncond_scale,
                uncond_tau=uncond_tau,
                tau_scale=tau_scale,
                n_ddim_steps=n_ddim_steps,
                ddim_eta=ddim_eta,
                ddim_spacing=ddim_spacing,
            )

            # Undo latent normalization (identity unless latent_norm was given)
            xt = self.ddpm.denormalize(xt, tau_t)

            if self.forecasting_type == "velocity":
                data_o["mu"] += [xt.reshape(batch_size, n_samples, 1, self.z_dim) + data_o["mu"][-1]]
            elif self.forecasting_type == "center":
                data_o["mu"] += [xt.reshape(batch_size, n_samples, 1, self.z_dim)]

        data_o["mu"] = torch.cat(data_o["mu"][1:], dim=-2)
        data_o["mu"] = data_o["mu"].reshape(-1, self.z_dim)

        data_o.update(self.decode(data_o["mu"], None))

        data_o = {
            k: v.reshape((batch_size, n_samples, n_future_windows) + v.shape[1:])
            for k, v in data_o.items()
        }

        return data_o

    def encode(self, data):
        B, T, W, J, _ = data["x6d"].shape
        if self.arena_size is not None:
            norm_root = self.normalize_root(data["root"])

            x_in = torch.cat((data["x6d"].view((B, T, W, -1)), norm_root), axis=-1)
        else:
            x_in = data["x6d"]

        param1, param2 = self.encoder(
            x_in.moveaxis(-2, -1).view(-1, self.in_channels, W)
        )
        param1 = param1.reshape(
            (
                B,
                T,
            )
            + param1.shape[1:]
        )
        param2 = param2.reshape(
            (
                B,
                T,
            )
            + param2.shape[1:]
        )

        data_o = {}
        data_o[self.dist_params[0]], data_o[self.dist_params[1]] = (
            param1[:, 1],
            param2[:, 1],
        )

        if self.prior == "beta":
            # Renormalizing to be between (-1, 1)
            mu = (param1 - 1 + 1e-8) / (param1 + param2 - 2 + 2e-8) * 2 - 1

            data_o["mu"] = mu[:, 1]
            data_o["z_past"] = mu[:, 0]
        else:
            data_o["z_past"] = param1[:, 0]

        return data_o


class ConditionalDiffusion(VAE):
    def __init__(
        self,
        in_channels,
        ch=[64, 128, 256, 512, 1024],
        kernel=5,
        z_dim=64,
        window=200,
        activation="prelu",
        is_diag=False,
        conditional_dim=0,
        init_dilation=None,
        disentangle=None,
        kinematic_tree=None,
        arena_size=None,
        disentangle_keys=None,
        conditional_keys=None,
        discrete_classes=None,
        prior="gaussian",
        diffusion_cond_keys=None,
        diffusion_cond_dim=None,
        cfg_prob=0,
        diffusion_type="e_pred",
        latent_norm=None,
        bottleneck_past=True,
        expand_latent=None,
        condition_tau=False,
        cfg_prob_tau=0.0,
    ):
        """
        expand_latent : [channels, length] or None
            Denoiser layout (see build_denoiser). None keeps the original U-Net.
        condition_tau : bool
            Condition the denoiser on the forecast horizon tau via one more
            conditioning token (see TauEmbedding), replaced by a learned null
            token for a random cfg_prob_tau fraction of training samples
            (independently of the past's cfg_prob), so tau can be guided. With
            bottleneck_past, z_past then becomes a single token beside it.
        bottleneck_past : bool
            If True, the denoiser is conditioned on a single z_past vector from
            the past_encoder's bottleneck. If False, the past_encoder skips its
            bottleneck and the denoiser cross-attends to a sequence of past
            tokens (one per time step of past_encoder's final feature map);
            extra diffusion_cond_keys are projected to one more token.
        """
        super().__init__(prior=prior)
        self.in_channels = in_channels
        self.ch = ch
        self.window = window
        self.is_diag = is_diag
        self.conditional_dim = conditional_dim
        self.kinematic_tree = kinematic_tree
        self.register_buffer("arena_size", arena_size)
        self.disentangle_keys = disentangle_keys
        self.conditional_keys = conditional_keys
        self.discrete_classes = discrete_classes
        self.diffusion_cond_keys = diffusion_cond_keys
        self.diffusion_cond_dim = diffusion_cond_dim
        self.cfg_prob = cfg_prob
        self.null_token = None
        self.z_dim = z_dim
        self.is_forecasting = True
        self.bottleneck_past = bottleneck_past
        self.condition_tau = condition_tau
        self.cfg_prob_tau = cfg_prob_tau
        # Conditioning is a sequence of z_dim tokens (past tokens or the single
        # z_past, extra variables, tau) unless it is the original single vector
        self.use_tokens = (not bottleneck_past) or condition_tau

        self.ddpm = DenoiseDiffusion(
            n_steps=1000,
            latent_dim=z_dim,
            diffusion_type=diffusion_type,
            latent_norm=latent_norm,
        )

        self.unet = build_denoiser(
            z_dim,
            d_cond=z_dim if self.use_tokens else z_dim + diffusion_cond_dim,
            expand_latent=expand_latent,
        )

        self.past_encoder = ResidualEncoder(
            in_channels,
            ch=ch,
            kernel=kernel,
            z_dim=z_dim,
            window=window,
            activation=activation,
            is_diag=is_diag,
            prior=None,
            init_dilation=init_dilation,
            bottleneck=bottleneck_past,
        )
        if self.use_tokens and diffusion_cond_dim > 0:
            self.cond_proj = nn.Linear(diffusion_cond_dim, z_dim)
        if condition_tau:
            self.tau_embed = TauEmbedding(z_dim)
            self.null_tau_token = nn.Parameter(0.02 * torch.randn(1, z_dim))

        self.encoder = ResidualEncoder(
            in_channels,
            ch=ch,
            kernel=kernel,
            z_dim=z_dim,
            window=window,
            activation=activation,
            is_diag=is_diag,
            prior=prior,
            init_dilation=init_dilation,
        )

        self.decoder = ResidualDecoder(
            in_channels,
            ch=ch,
            kernel=kernel,
            z_dim=z_dim,
            window=window,
            activation=activation,
            conditional_dim=conditional_dim,
            init_dilation=init_dilation,
        )

        # Scrubbers
        if disentangle is not None:
            self.disentangle = nn.ModuleDict()
            for k, v in disentangle.items():
                self.disentangle[k] = nn.ModuleDict(v)
        else:
            self.disentangle = nn.ModuleDict(nn.ModuleDict())

        self.mi_estimator = None

    def normalize_root(self, root):
        norm_root = root - self.arena_size[0]
        norm_root = 2 * norm_root / (self.arena_size[1] - self.arena_size[0]) - 1
        return norm_root

    def inv_normalize_root(self, norm_root):
        root = 0.5 * (norm_root + 1) * (self.arena_size[1] - self.arena_size[0])
        root += self.arena_size[0]
        return root

    def tau_token(self, tau: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        """(B, 1, z_dim) tau token; the learned null token if tau is None, and
        for a random cfg_prob_tau fraction of samples during training"""
        null = self.null_tau_token.expand(batch_size, -1)
        if tau is None:
            return null[:, None]
        token = self.tau_embed(tau)
        if self.training and self.cfg_prob_tau > 0:
            mask = torch.rand(batch_size, device=token.device) < self.cfg_prob_tau
            token = torch.where(mask[:, None], null, token)
        return token[:, None]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        rename_legacy_cond_keys(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def build_cond(
        self,
        z_past: torch.Tensor,
        data: Optional[Dict] = None,
        tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Denoiser conditioning from z_past -- (B, z_dim) if bottleneck_past, else
        (B, n_tokens, z_dim) -- the extra conditioning variables
        diffusion_cond_keys read from data (None -> null variables; see
        build_cond_vars) and, if condition_tau, the horizon tau (B,) (None ->
        null tau): concatenated into one vector (bottleneck_past without
        condition_tau), else a token sequence -- the past token(s), one
        projected token for the extra variables and one tau token.
        """
        cond_vars = build_cond_vars(self, data, z_past.shape[0], z_past.device)
        if not self.use_tokens:
            return torch.cat([z_past] + cond_vars, dim=-1)
        tokens = z_past if z_past.dim() == 3 else z_past[:, None]
        if len(cond_vars) > 0:
            cond_token = self.cond_proj(torch.cat(cond_vars, dim=-1))
            tokens = torch.cat([tokens, cond_token[:, None]], dim=1)
        if self.condition_tau:
            tokens = torch.cat([tokens, self.tau_token(tau, tokens.shape[0])], dim=1)
        return tokens

    def get_null_past(self):
        """past_encoder output for an all-zero (null) past window: (1, z_dim),
        or (1, n_tokens, z_dim) if not bottleneck_past"""
        device = next(self.encoder.parameters()).device
        return self.past_encoder(
            torch.zeros((1, self.in_channels, self.window), device=device)
        )

    def get_null(self):
        return self.build_cond(self.get_null_past(), None)

    def get_samples(
        self,
        data,
        n_future_windows,
        n_samples,
        uncond_scale,
        tau: int = 1,
        tau_scale: float = 1,
        n_ddim_steps: Optional[int] = None,
        ddim_eta: float = 0.0,
        ddim_spacing: str = "quadratic",
    ):
        """
        Autoregressive forecast: each of n_future_windows steps predicts the latent
        tau windows (tau * n_skip frames) after the previous one.
        uncond_scale guides the past; tau_scale guides tau (condition_tau
        models only; 1 = ordinary conditioning on tau).
        n_ddim_steps=None samples each step with DDPM over all timesteps
        (minus skip_steps); n_ddim_steps=k uses DDIM with k timesteps, noise
        level ddim_eta and ddim_spacing (see DenoiseDiffusion.sample).
        """
        batch_size = data["x6d"].shape[0]
        tau_t = torch.full(
            (batch_size * n_samples,), tau, dtype=torch.long, device=data["x6d"].device
        )

        data_o = self.encode({k: data[k] for k in ["x6d", "root"]})
        # Extra conditioning variables, in the same (batch-major, sample-minor)
        # order as z_past
        cond_data = {
            k: data[k].repeat_interleave(n_samples, dim=0)
            for k in (self.diffusion_cond_keys or [])
        }

        data_o.update(self.decode(data_o["mu"]))
        # (batch_size * n_samples, z_dim) or (batch_size * n_samples, n_tokens, z_dim)
        z_past = data_o["z_past"].repeat_interleave(n_samples, dim=0)
        # data_o = {
        #     k: data_o[k].reshape(batch_size, 1, 1, *data_o[k].shape[1:])
        #     for k in ["mu", "x6d", "root"]
        # }
        data_o = {
            k: [] for k in ["mu", "x6d", "root"]
        }  # Batch x n_samples x steps_ahead x ...
        null_past = self.get_null_past()
        null_past = null_past.expand((batch_size * n_samples,) + null_past.shape[1:])
        uncond = self.build_cond(null_past, None, tau_t)
        uncond_tau = (
            self.build_cond(null_past, None, None) if self.condition_tau else None
        )
        for t_step in range(1, n_future_windows + 1):
            cond = self.build_cond(z_past, cond_data, tau_t)

            xt = torch.randn(
                (batch_size * n_samples, self.z_dim), device=z_past.device
            )
            xt = self.ddpm.sample(
                self.unet,
                xt,
                cond,
                uncond=uncond,
                uncond_scale=uncond_scale,
                uncond_tau=uncond_tau,
                tau_scale=tau_scale,
                n_ddim_steps=n_ddim_steps,
                ddim_eta=ddim_eta,
                ddim_spacing=ddim_spacing,
            )

            # Undo latent normalization (identity unless latent_norm was given)
            xt = self.ddpm.denormalize(xt, tau_t)

            # print(xt[0])
            data_temp = {"mu": xt.clone().detach()}
            data_temp.update(self.decode(data_temp["mu"], None))
            data_temp = process_batch_output(
                data_temp, self.kinematic_tree, data["offsets"]
            )
            if t_step != n_future_windows:
                z_past = self.past_encode(data_temp).clone()

            for k in data_o.keys():
                if k in data_temp.keys():
                    data_o[k] += [
                        data_temp[k].reshape(
                            batch_size, n_samples, 1, *data_temp[k].shape[1:]
                        )
                    ]

        data_o = {k: torch.cat(data_o[k], dim=2) for k in data_o.keys()}
        # data_o.update(self.decode(data_o["mu"], None))
        return data_o

    def past_encode(self, data):
        save_shape = data["x6d"].shape[:-2]
        if self.arena_size is not None:
            norm_root = self.normalize_root(data["root"])

            x_in = torch.cat((data["x6d"].view(*save_shape, -1), norm_root), axis=-1)
        else:
            x_in = data["x6d"]

        z_past = self.past_encoder(
            x_in.moveaxis(-2, -1).view(-1, self.in_channels, self.window)
        )
        # (..., z_dim), or (..., n_tokens, z_dim) if not bottleneck_past
        z_past = z_past.reshape(*save_shape[:-1], *z_past.shape[1:])

        return z_past

    def pure_encode(self, data):
        B, T = data["x6d"].shape[:2]
        if self.arena_size is not None:
            norm_root = self.normalize_root(data["root"])

            x_in = torch.cat(
                (data["x6d"].view(data["x6d"].shape[:-2] + (-1,)), norm_root), axis=-1
            )
        else:
            x_in = data["x6d"]

        x_in = x_in.moveaxis(-2, -1).view(-1, self.in_channels, self.window)

        data_o = {}
        data_o[self.dist_params[0]], data_o[self.dist_params[1]] = self.encoder(x_in)

        if self.prior == "beta":
            # Renormalizing to be between (-1, 1)
            data_o["mu"] = (data_o["alpha"] - 1 + 1e-8) / (
                data_o["alpha"] + data_o["beta"] - 2 + 2e-8
            ) * 2 - 1

        data_o = {
            k: v.reshape(
                (
                    B,
                    T,
                )
                + v.shape[1:]
            )
            for k, v in data_o.items()
        }

        return data_o

    def encode(self, data):
        if self.arena_size is not None:
            norm_root = self.normalize_root(data["root"])

            x_in = torch.cat(
                (data["x6d"].view(data["x6d"].shape[:-2] + (-1,)), norm_root), axis=-1
            )
        else:
            x_in = data["x6d"]

        x_past = (
            x_in[:, 0, ...].moveaxis(-2, -1).view(-1, self.in_channels, self.window)
        )

        # Classifier-free conditional guidance
        if (self.cfg_prob is not None) and (self.cfg_prob > 0) and (self.training):
            mask = torch.rand(x_past.shape[0], device=x_in.device) < self.cfg_prob
            x_past[mask, ...] = 0

        data_o = {}
        if "vae_z" in data.keys():
            # Cached latent of window 1 from the frozen encoder (see
            # neurobe4.data.latents) -- skip the encoder
            data_o["mu"] = data["vae_z"][:, 1]
        else:
            x_future = (
                x_in[:, 1, ...].moveaxis(-2, -1).view(-1, self.in_channels, self.window)
            )
            data_o[self.dist_params[0]], data_o[self.dist_params[1]] = self.encoder(
                x_future
            )

            if self.prior == "beta":
                # Renormalizing to be between (-1, 1)
                data_o["mu"] = (data_o["alpha"] - 1 + 1e-8) / (
                    data_o["alpha"] + data_o["beta"] - 2 + 2e-8
                ) * 2 - 1

        data_o["z_past"] = self.past_encoder(x_past)

        return data_o

    def decode(self, z, data=None):
        data_o = {}

        x_hat = self.decoder(z).moveaxis(-1, 1)

        if self.arena_size is None:
            x6d = x_hat
        else:
            x6d = x_hat[..., :-3]
            data_o["root"] = self.inv_normalize_root(x_hat[..., -3:]).reshape(
                z.shape[0], self.window, 3
            )

        x6d = F.normalize(x6d.reshape(z.shape[0], 1, -1, 2, 3), dim=-1)
        data_o["x6d"] = x6d.reshape(z.shape[0], self.window, -1, 6)

        return data_o

    def forward(self, data):
        data_o = self.encode(data)

        # Reparameterize
        if self.prior == "gaussian":
            z = (
                self.sampling(data_o["mu"], data_o["L"])
                if self.training
                else data_o["mu"]
            )
        elif self.prior == "beta":
            beta_dist = torch.distributions.Beta(data_o["alpha"], data_o["beta"])
            data_o["beta_dist"] = beta_dist
            z = beta_dist.rsample() * 2 - 1

        data_o["z"] = z

        data_o.update(self.decode(z, data))

        return data_o


# Conditioning sources of SocialConditionalDiffusion: window tokens from the
# shared past_encoder; the partner sources also carry the relative pose
# (rel_pose, see neurobe4.data.train_utils.relative_pose) through a small MLP.
SOCIAL_SOURCES = ("own_past", "partner_past", "partner_concurrent")
SOCIAL_PARTNER_SOURCES = ("partner_past", "partner_concurrent")


class SocialConditionalDiffusion(ConditionalDiffusion):
    """
    ConditionalDiffusion for social forecasting (SocialDiskDatasetTPaired):
    the target animal's next window (window 1) is predicted from any subset of

        own_past            the target's window 0
        partner_past        the partner's window 0, plus rel_pose
        partner_concurrent  the partner's window 1 (same time as the target),
                            plus rel_pose

    rel_pose: the partner's dx, dy and cos, sin of its relative heading in
    the target's egocentric frame at the frame shared by windows 0 and 1
    (n_skip = window - 1).

    Each source is a group of conditioning tokens -- its window through the
    shared past_encoder and, for partner sources, one rel_pose token through
    an MLP, all plus a learned per-source role embedding -- with its own
    learned null token. There is no tau conditioning: the target is always
    the next window.

    One model covers every combination: during training each sample
    conditions on one subset drawn uniformly from social_subsets (all
    other sources null), and with probability cfg_prob on none (for
    guidance). At sampling time the given sources are chosen per call and
    guided as a nested chain (see get_samples).
    """

    def __init__(
        self,
        *args,
        social_sources=SOCIAL_SOURCES,
        social_subsets=None,
        **kwargs,
    ):
        """
        social_sources : list of str
            Sources (from SOCIAL_SOURCES) the model has.
        social_subsets : list of lists of str, optional
            Source combinations trained on, each drawn with equal probability
            (e.g. [[own_past], [partner_past], [partner_concurrent],
            [own_past, partner_past]]). Default: each source alone
            plus all of them together.
        Other arguments as ConditionalDiffusion; diffusion_cond_keys and
        condition_tau are not supported (cfg_prob is the probability of
        conditioning on nothing).
        """
        if kwargs.get("diffusion_cond_keys"):
            raise ValueError("SocialConditionalDiffusion does not support extra conditional_keys")
        if kwargs.get("condition_tau"):
            raise ValueError("SocialConditionalDiffusion does not support tau conditioning")
        super().__init__(*args, **kwargs)
        self.use_tokens = True

        unknown = set(social_sources) - set(SOCIAL_SOURCES)
        if unknown:
            raise ValueError(f"Unknown social sources {sorted(unknown)}; options: {SOCIAL_SOURCES}")
        self.social_sources = list(social_sources)
        if social_subsets is None:
            social_subsets = [[s] for s in self.social_sources] + [self.social_sources]
        for subset in social_subsets:
            if not set(subset) <= set(self.social_sources):
                raise ValueError(f"Subset {subset} uses sources the model does not have")
        self.social_subsets = [list(s) for s in social_subsets]
        # (n_subsets, n_sources) membership, for sampling subsets per sample
        self.register_buffer(
            "subset_mask",
            torch.tensor(
                [[s in subset for s in self.social_sources] for subset in self.social_subsets],
                dtype=torch.bool,
            ),
            persistent=False,
        )

        z_dim = self.z_dim
        self.role_embed = nn.ParameterDict(
            {s: nn.Parameter(0.02 * torch.randn(1, 1, z_dim)) for s in self.social_sources}
        )
        self.null_tokens = nn.ParameterDict(
            {s: nn.Parameter(0.02 * torch.randn(1, 1, z_dim)) for s in self.social_sources}
        )
        if any(s in SOCIAL_PARTNER_SOURCES for s in self.social_sources):
            # (dx, dy, cos, sin of relative heading) -> one token, shared by
            # the partner sources (which differ by role embedding)
            self.rel_mlp = nn.Sequential(
                nn.Linear(4, z_dim), nn.SiLU(), nn.Linear(z_dim, z_dim)
            )
            # Distances are scaled by the arena's half-width
            rel_scale = (
                float(self.arena_size[1, 0] - self.arena_size[0, 0]) / 2
                if self.arena_size is not None
                else 1.0
            )
            self.register_buffer("rel_scale", torch.tensor(rel_scale))

    def _window_input(self, x6d: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
        """(B, T, W, K, 6), (B, T, W, 3) -> past_encoder input (B, T, C, W)"""
        x_in = x6d.reshape(x6d.shape[:-2] + (-1,))
        if self.arena_size is not None:
            x_in = torch.cat((x_in, self.normalize_root(root)), dim=-1)
        return x_in.moveaxis(-2, -1)

    def _rel_token(self, rel_pose: torch.Tensor) -> torch.Tensor:
        """(B, 4) relative pose (dx, dy, cos, sin) -> (B, 1, z_dim) token
        (before role embedding); dx, dy scaled by the arena half-width"""
        feat = torch.cat([rel_pose[:, :2] / self.rel_scale, rel_pose[:, 2:]], dim=-1)
        return self.rel_mlp(feat)[:, None]

    def encode_sources(self, data: Dict) -> Dict[str, torch.Tensor]:
        """Conditioning tokens of every source the model has for target
        window 1: {source: (B, n_tokens, z_dim)}; partner sources end with
        their rel_pose token"""
        tokens = {}
        windows = []  # (source, (B, C, W)), through one past_encoder call
        if "own_past" in self.social_sources:
            windows.append(("own_past", self._window_input(data["x6d"][:, :1], data["root"][:, :1])[:, 0]))
        if "partner_past" in self.social_sources or "partner_concurrent" in self.social_sources:
            partner = self._window_input(data["x6d_partner"][:, :2], data["root_partner"][:, :2])
            if "partner_past" in self.social_sources:
                windows.append(("partner_past", partner[:, 0]))
            if "partner_concurrent" in self.social_sources:
                windows.append(("partner_concurrent", partner[:, 1]))
        if windows:
            z = self.past_encoder(torch.cat([w for _, w in windows], dim=0))
            if z.dim() == 2:  # bottleneck_past: one token per window
                z = z[:, None]
            rel = (
                self._rel_token(data["rel_pose"])
                if any(s in SOCIAL_PARTNER_SOURCES for s in self.social_sources)
                else None
            )
            for (source, _), tok in zip(windows, z.chunk(len(windows))):
                if source in SOCIAL_PARTNER_SOURCES:
                    tok = torch.cat([tok, rel], dim=1)
                tokens[source] = tok + self.role_embed[source]
        return tokens

    def encode(self, data):
        """Target latent of window 1 (cached vae_z or the encoder) and the
        conditioning tokens of every source, under "z_past" as a dict"""
        data_o = {}
        if "vae_z" in data.keys():
            data_o["mu"] = data["vae_z"][:, 1]
        else:
            x_future = self._window_input(data["x6d"][:, 1:2], data["root"][:, 1:2])[:, 0]
            data_o[self.dist_params[0]], data_o[self.dist_params[1]] = self.encoder(x_future)
            if self.prior == "beta":
                data_o["mu"] = (data_o["alpha"] - 1 + 1e-8) / (
                    data_o["alpha"] + data_o["beta"] - 2 + 2e-8
                ) * 2 - 1
        data_o["z_past"] = self.encode_sources(data)
        return data_o

    def sample_present(self, batch_size: int, device) -> torch.Tensor:
        """(B, n_sources) bool: a uniformly drawn social_subsets entry per
        sample, emptied for a random cfg_prob fraction"""
        idx = torch.randint(len(self.social_subsets), (batch_size,), device=device)
        present = self.subset_mask[idx]
        if self.cfg_prob:
            drop = torch.rand(batch_size, device=device) < self.cfg_prob
            present = present & ~drop[:, None]
        return present

    def build_cond(
        self,
        z_past: Dict[str, torch.Tensor],
        data: Optional[Dict] = None,
        tau: Optional[torch.Tensor] = None,
        sources=None,
    ) -> torch.Tensor:
        """
        Token sequence (B, n_tokens, z_dim): every source's tokens in
        social_sources order -- its learned null token(s) where the source is
        absent.

        z_past : {source: (B, n, z_dim)} from encode / encode_sources
        sources : list of present sources for every sample, or None to draw
            a subset per sample (sample_present; the training loss)
        data, tau : unused (no extra conditioning variables or tau), accepted
            for the shared loss interface
        """
        B = next(iter(z_past.values())).shape[0]
        device = next(iter(z_past.values())).device
        if sources is None:
            present = self.sample_present(B, device)
        else:
            unknown = set(sources) - set(self.social_sources)
            if unknown:
                raise ValueError(f"Model has no sources {sorted(unknown)}")
            present = torch.tensor(
                [s in sources for s in self.social_sources], device=device
            ).expand(B, -1)

        tokens = []
        for j, s in enumerate(self.social_sources):
            null = self.null_tokens[s].expand_as(z_past[s])
            tokens.append(torch.where(present[:, j, None, None], z_past[s], null))
        return torch.cat(tokens, dim=1)

    def get_null(self):
        raise NotImplementedError("use build_cond(z_past, sources=[]) for the null conditioning")

    def get_samples(
        self,
        data,
        n_samples,
        guidance,
        n_ddim_steps: Optional[int] = None,
        ddim_eta: float = 0.0,
        ddim_spacing: str = "quadratic",
    ):
        """
        Forecast the target's next window (window 1) from window 0.

        data : prepare_batch output of SocialDiskDatasetTPaired (windows 0
            and 1 of both animals and rel_pose).
        guidance : list of (source, scale), source in social_sources -- the
            given conditions, guided as the nested chain
                eps = e(null) + s_1 [e(c_1) - e(null)]
                              + s_2 [e(c_1 + c_2) - e(c_1)] + ...
            in list order. Sources not listed are null. Examples:
                [("own_past", 5)]                       case 1
                [("partner_past", 5)]                   case 2
                [("partner_concurrent", 5)]             case 3
                [("own_past", 5), ("partner_past", 2)]  case 4

        Returns {"mu", "x6d", "root"}: (B, n_samples, 1, ...)
        """
        names = [g[0] for g in guidance]
        unknown = set(names) - set(self.social_sources)
        if unknown:
            raise ValueError(f"Model has no sources {sorted(unknown)}")
        scales = [g[1] for g in guidance]
        chain = [[]]
        for name in names:
            chain.append(chain[-1] + [name])

        batch_size = data["x6d"].shape[0]
        device = data["x6d"].device
        z_past = {
            k: v.repeat_interleave(n_samples, dim=0)
            for k, v in self.encode_sources(data).items()
        }
        conds = [self.build_cond(z_past, sources=c) for c in chain]
        xt = torch.randn((batch_size * n_samples, self.z_dim), device=device)
        xt = self.ddpm.sample(
            self.unet,
            xt,
            conds,
            uncond_scale=scales,
            n_ddim_steps=n_ddim_steps,
            ddim_eta=ddim_eta,
            ddim_spacing=ddim_spacing,
        )
        xt = self.ddpm.denormalize(xt)
        data_o = {"mu": xt}
        data_o.update(self.decode(xt, None))
        data_o = process_batch_output(data_o, self.kinematic_tree, data["offsets"])
        return {
            k: data_o[k].reshape(batch_size, n_samples, 1, *data_o[k].shape[1:])
            for k in ["mu", "x6d", "root"]
            if k in data_o
        }


class ConditionalForecaster(VAE):
    def __init__(
        self,
        in_channels,
        ch=[64, 128, 256, 512, 1024],
        kernel=5,
        z_dim=64,
        window=200,
        activation="prelu",
        is_diag=False,
        conditional_dim=0,
        init_dilation=None,
        disentangle=None,
        kinematic_tree=None,
        arena_size=None,
        disentangle_keys=None,
        conditional_keys=None,
        discrete_classes=None,
        prior="gaussian",
    ):
        super().__init__(prior=prior)
        self.in_channels = in_channels
        self.ch = ch
        self.window = window
        self.is_diag = is_diag
        self.conditional_dim = conditional_dim
        self.kinematic_tree = kinematic_tree
        self.register_buffer("arena_size", arena_size)
        self.disentangle_keys = disentangle_keys
        self.conditional_keys = conditional_keys
        self.discrete_classes = discrete_classes
        self.is_forecasting = True

        self.past_encoder = ResidualEncoder(
            in_channels,
            ch=ch,
            kernel=kernel,
            z_dim=z_dim,
            window=window,
            activation=activation,
            is_diag=is_diag,
            prior=None,
            init_dilation=init_dilation,
        )

        self.future_encoder = ResidualEncoder(
            in_channels,
            ch=ch,
            kernel=kernel,
            z_dim=z_dim,
            window=window,
            activation=activation,
            is_diag=is_diag,
            prior=prior,
            init_dilation=init_dilation,
        )

        self.decoder = ResidualDecoder(
            in_channels,
            ch=ch,
            kernel=kernel,
            z_dim=z_dim,
            window=window,
            activation=activation,
            conditional_dim=conditional_dim,
            init_dilation=init_dilation,
        )

        # Scrubbers
        if disentangle is not None:
            self.disentangle = nn.ModuleDict()
            for k, v in disentangle.items():
                self.disentangle[k] = nn.ModuleDict(v)
        else:
            self.disentangle = nn.ModuleDict(nn.ModuleDict())

        self.mi_estimator = None

    def normalize_root(self, root):
        norm_root = root - self.arena_size[0]
        norm_root = 2 * norm_root / (self.arena_size[1] - self.arena_size[0]) - 1
        return norm_root

    def inv_normalize_root(self, norm_root):
        root = 0.5 * (norm_root + 1) * (self.arena_size[1] - self.arena_size[0])
        root += self.arena_size[0]
        return root

    def encode(self, data):
        if self.arena_size is not None:
            norm_root = self.normalize_root(data["root"])

            x_in = torch.cat(
                (data["x6d"].view(data["x6d"].shape[:-2] + (-1,)), norm_root), axis=-1
            )
        else:
            x_in = data["x6d"]

        x_past = (
            x_in[:, 0, ...].moveaxis(-2, -1).view(-1, self.in_channels, self.window)
        )
        x_future = (
            x_in[:, 1, ...].moveaxis(-2, -1).view(-1, self.in_channels, self.window)
        )
        data_o = {}
        data_o[self.dist_params[0]], data_o[self.dist_params[1]] = self.future_encoder(
            x_future
        )
        data_o["z_past"] = self.past_encoder(x_past)

        if self.prior == "beta":
            # Renormalizing to be between (-1, 1)
            data_o["mu"] = (data_o["alpha"] - 1 + 1e-8) / (
                data_o["alpha"] + data_o["beta"] - 2 + 2e-8
            ) * 2 - 1

        return data_o

    def decode(self, z, data):
        data_o = {}
        if self.conditional_dim > 0:
            # Concatenating disentanglement variables together
            data_o["var"] = [
                (
                    F.one_hot(data[k].ravel().long(), len(self.discrete_classes[k]))
                    if k in self.discrete_classes.keys()
                    else data[k]
                )
                for k in self.conditional_keys
            ]
            data_o["var"] = torch.cat(data_o["var"], dim=-1)
            z = torch.cat(
                [z, data_o["var"]],
                dim=-1,
            )

        x_hat = self.decoder(z).moveaxis(-1, 1)

        if self.arena_size is None:
            x6d = x_hat
        else:
            x6d = x_hat[..., :-3]
            data_o["root"] = self.inv_normalize_root(x_hat[..., -3:]).reshape(
                z.shape[0], self.window, 3
            )

        x6d = F.normalize(x6d.reshape(z.shape[0], 1, -1, 2, 3), dim=-1)
        data_o["x6d"] = x6d.reshape(z.shape[0], self.window, -1, 6)

        return data_o

    def forward(self, data):
        data_o = self.encode(data)

        # Reparameterize
        if self.prior == "gaussian":
            z = (
                self.sampling(data_o["mu"], data_o["L"])
                if self.training
                else data_o["mu"]
            )
        elif self.prior == "beta":
            beta_dist = torch.distributions.Beta(data_o["alpha"], data_o["beta"])
            data_o["beta_dist"] = beta_dist
            z = beta_dist.rsample() * 2 - 1

        data_o["z"] = z + data_o["z_past"]

        data_o.update(self.decode(z, data))

        # Running disentangle
        data_o["disentangle"] = {}
        if "linear" in self.disentangle.keys():
            data_o["disentangle"]["linear"] = {
                k: model(data_o["mu"])
                for k, model in self.disentangle["linear"].items()
            }

        # Forward pass through all scrubbers if necessary
        for method, module_dict in self.disentangle.items():
            if method == "linear":
                ## Placeholder for if we reimplement in the future
                continue
            else:
                data_o["disentangle"][method] = {}
                for k, model in module_dict.items():
                    if "linear" in self.disentangle.keys():
                        latent = data_o["disentangle"]["linear"][k]["z_null"]
                    else:
                        latent = data_o["mu"]
                    if method == "adversarial_net":
                        data_o["disentangle"][method][k] = model(latent, data_o["var"])
                    else:
                        data_o["disentangle"][method][k] = model(latent)

        return data_o


class ForecastingVAE(ResVAE):
    def __init__(
        self,
        in_channels,
        ch=[64, 128, 256, 512, 1024],
        kernel=5,
        z_dim=64,
        window=200,
        activation="prelu",
        is_diag=False,
        conditional_dim=0,
        init_dilation=None,
        disentangle=None,
        kinematic_tree=None,
        arena_size=None,
        disentangle_keys=None,
        conditional_keys=None,
        discrete_classes=None,
        prior="gaussian",
    ):
        super().__init__(
            in_channels=in_channels,
            ch=ch,
            kernel=kernel,
            z_dim=z_dim,
            window=window,
            activation=activation,
            is_diag=is_diag,
            conditional_dim=conditional_dim,
            init_dilation=init_dilation,
            disentangle=disentangle,
            kinematic_tree=kinematic_tree,
            arena_size=arena_size,
            disentangle_keys=disentangle_keys,
            conditional_keys=conditional_keys,
            discrete_classes=discrete_classes,
            prior=prior,
        )
        self.is_forecasting = True

    def encode(self, data):
        if self.arena_size is not None:
            norm_root = self.normalize_root(data["root"][:, 0, ...])

            x_in = torch.cat(
                (
                    data["x6d"].view(data["x6d"].shape[:-2] + (-1,))[:, 0, ...],
                    norm_root,
                ),
                axis=-1,
            )
        else:
            x_in = data["x6d"]

        # x_in = x_in.moveaxis(-2, -1).view(-1, self.in_channels, self.window)
        # x_future = x_in[:,1,...].moveaxis(-2, -1).view(-1, self.in_channels, self.window)
        data_o = {}
        data_o[self.dist_params[0]], data_o[self.dist_params[1]] = self.encoder(
            x_in.moveaxis(-2, -1).view(-1, self.in_channels, self.window)
        )
        # data_o["z_past"] = self.encoder(x_)

        if self.prior == "beta":
            # Renormalizing to be between (-1, 1)
            data_o["mu"] = (data_o["alpha"] - 1 + 1e-8) / (
                data_o["alpha"] + data_o["beta"] - 2 + 2e-8
            ) * 2 - 1

        return data_o

    def get_samples(self, data, n_future_windows, **kwargs):
        B = data["x6d"].shape[0]
        data_o = {"x6d": [], "root": [], "mu": []}
        for t_step in range(n_future_windows):
            if t_step == 0:
                data_step = self.encode(data)
                z = self.sampling(data_step["mu"], data_step["L"])
                data_step.update(self.decode(z, data))
            else:
                data_step = self.encode({k: v[-1] for k, v in data_o.items()})
                z = self.sampling(data_step["mu"], data_step["L"])
                data_step.update(self.decode(z, data_o))

            data_step = process_batch_output(
                data_step, kinematic_tree=self.kinematic_tree, offsets=data["offsets"]
            )
            data_o = {
                k: v + [data_step[k].reshape(B, 1, *data_step[k].shape[1:])]
                for k, v in data_o.items()
            }

        data_o = {k: torch.cat(v, dim=1) for k, v in data_o.items()}
        return data_o
