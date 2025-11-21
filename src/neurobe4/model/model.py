from scrubvae.model.residual import (
    VAE,
    ResidualEncoder,
    ResidualDecoder,
    ResVAE,
    CholeskyL,
)

from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from neurobe4.model.unet_sd import UNetModel

# from labml_nn.diffusion.ddpm.utils import gather


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


class DenoiseDiffusion(nn.Module):
    """
    ## Denoise Diffusion
    """

    def __init__(self, n_steps: int):
        """
        * `eps_model` is $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ model
        * `n_steps` is $t$
        * `device` is the device to place constants on
        """
        super().__init__()

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

    def get_eps(
        self,
        eps_model,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor = None,
        uncond_scale: float = 1,
    ):
        if uncond is None or uncond_scale == 1.0:
            return eps_model(xt, t, cond)

        # Duplicate $x_t$ and $t$
        x_in = torch.cat([xt] * 2, dim=0)
        t_in = torch.cat([t] * 2, dim=0)
        # Concatenated $c$ and $c_u$
        c_in = torch.cat([uncond, cond], dim=0)
        # Get $\epsilon_\text{cond}(x_t, c)$ and $\epsilon_\text{cond}(x_t, c_u)$
        e_t_uncond, e_t_cond = eps_model(x_in, t_in, c_in).chunk(2)
        # Calculate
        # $$\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$$
        e_t = e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)

        #
        return e_t

    def p_sample(
        self,
        eps_model,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor = None,
        uncond_scale: float = 1,
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
        eps_theta = self.get_eps(eps_model, xt.squeeze(), t, cond, uncond, uncond_scale)
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
        # Sample
        return mean + (var**0.5) * eps

    def loss(
        self,
        eps_model,
        x0: torch.Tensor,
        cond: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        return_mean=False,
        uncond: torch.Tensor = None,
        uncond_scale: float = 1,
    ):
        """
        #### Simplified Loss

        $$L_{\text{simple}}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """
        # Get batch size
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
        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        eps_theta = self.get_eps(eps_model, xt.squeeze(), t, cond, uncond, uncond_scale)

        # MSE loss
        if return_mean:
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
            # Sample
            return F.mse_loss(noise, eps_theta), mean + (var**0.5) * eps
        else:
            return F.mse_loss(noise, eps_theta)

    def t0_time_consistency_loss(
        self,
        eps_model,
        x0: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor = None,
        uncond_scale: float = 1,
    ):
        # Get batch size
        x0 = x0[:, None, :]
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        t = torch.zeros((batch_size,), device=x0.device, dtype=torch.long)

        # $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
        noise = torch.randn_like(x0)

        # Sample $x_t$ for $q(x_t|x_0)$
        xt = self.q_sample(x0, t, eps=noise)
        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        eps_theta = self.get_eps(eps_model, xt.squeeze(), t, cond, uncond, uncond_scale)

        # MSE loss
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
        # Sample
        return mean + (var**0.5) * eps

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
        self.kinematic_tree=kinematic_tree


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
            x_in.squeeze()
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

        data_o["x6d"] = x6d.reshape(z.shape[0], 1, -1, 6)

        return data_o



class MLPDiff(MLPVAE):
    def __init__(
        self,
        in_dim,
        layer_dims=[128, 256, 512, 1024],
        window=51,
        z_dim=64,
        activation="prelu",
        is_diag=False,
        prior="gaussian",
        arena_size=None,
        kinematic_tree=None,
        diffusion_cond_keys=None,
        diffusion_cond_dim=None,
        cfg_prob=0,
    ):
        super().__init__(
            in_dim=in_dim,
            z_dim=z_dim,
            window=window,
            activation=activation,
            is_diag=is_diag,
            kinematic_tree=kinematic_tree,
            arena_size=arena_size,
            prior=prior,
        )
        self.ddpm = DenoiseDiffusion(n_steps=1000)
        self.diffusion_cond_keys = diffusion_cond_keys
        self.diffusion_cond_dim = diffusion_cond_dim
        self.cfg_prob = cfg_prob
        self.null_token = None

        self.past_encoder = ResidualEncoder(
            in_channels=z_dim,
            ch=layer_dims,
            kernel=3,
            z_dim=z_dim,
            window=window,
            activation=activation,
            is_diag=is_diag,
            prior=None,
            init_dilation=None,
        )
        self.unet = UNetModel(
            in_channels=1,
            out_channels=1,
            channels=z_dim,
            attention_levels=[0, 1],
            n_res_blocks=1,
            channel_multipliers=[1, 2, 2],
            n_heads=4,
            tf_layers=1,
            d_cond=z_dim + diffusion_cond_dim,
        )

    def get_null(self):
        device = next(self.encoder.parameters()).device
        null_past = torch.zeros((1, self.diffusion_cond_dim))
        null_past = self.encoder(
            torch.zeros((1, self.in_channels, self.window), device=device)
        )[0]
        other_null = torch.zeros((1, self.diffusion_cond_dim), device=device)
        null_token = torch.cat([null_past, other_null], dim=-1)

        return null_token

    def get_samples(self, data, n_steps_ahead, n_samples, uncond_scale, skip_steps):
        batch_size = data["x6d"].shape[0]

        if self.null_token is None:
            self.null_token = self.get_null()

        past_data = {k: v[:, 0] for k, v in data.items() if k in ["x6d", "root"]}
        cond_var = []
        for k in self.diffusion_cond_keys:
            cond_temp = (
                F.one_hot(data[k].ravel().long(), len(self.discrete_classes[k]))
                if k in self.discrete_classes.keys()
                else data[k]
            )
            cond_var += [cond_temp]

        cond_var = torch.cat(cond_var, dim=-1).repeat((n_samples, 1))
        norm_root = self.normalize_root(past_data["root"])

        x_in = torch.cat(
            (past_data["x6d"].view(past_data["x6d"].shape[:-2] + (-1,)), norm_root),
            axis=-1,
        )
        uncond = self.null_token.repeat(batch_size * n_samples, 1)
        init_mu = self.encoder(x_in.moveaxis(-2, -1))[0].reshape(
            batch_size, 1, 1, self.z_dim
        )
        data_o = {"mu": [init_mu.repeat(1, n_samples, 1, 1)]}
        for t_step in range(1, n_steps_ahead + 1):
            # import pdb; pdb.set_trace()
            cond = torch.cat(
                [data_o["mu"][-1].reshape(-1, self.z_dim), cond_var], dim=-1
            )
            xt = torch.randn_like(data_o["mu"][-1].reshape(-1, self.z_dim))
            time_steps = torch.flip(
                torch.arange(
                    self.ddpm.n_steps, dtype=int, device=past_data["x6d"].device
                ),
                dims=(0,),
            )[skip_steps:]
            for _, step in enumerate(time_steps):
                # import pdb; pdb.set_trace()
                t = xt.new_full((batch_size * n_samples,), step, dtype=torch.long)
                try:
                    xt = self.ddpm.p_sample(
                        self.unet, xt, t, cond, uncond=uncond, uncond_scale=uncond_scale
                    ).squeeze()  ##TODO: Debug multiple samples
                except:
                    import pdb

                    pdb.set_trace()

            data_o["mu"] += [xt.reshape(batch_size, n_samples, 1, self.z_dim)]

        # import pdb; pdb.set_trace()
        data_o["mu"] = torch.cat(data_o["mu"], dim=-2)
        data_o["mu"] = data_o["mu"].reshape(-1, self.z_dim)

        data_o.update(self.decode(data_o["mu"], None))

        data_o = {
            k: v.reshape((batch_size, n_samples, n_steps_ahead + 1) + v.shape[1:])
            for k, v in data_o.items()
        }

        return data_o

    def encode(self, data):
        import pdb; pdb.set_trace()
        B, T, W, J, _ = data["x6d"].shape
        if self.arena_size is not None:
            norm_root = self.normalize_root(data["root"])

            x_in = torch.cat((data["x6d"].view((B, T, W, -1)), norm_root), axis=-1)
        else:
            x_in = data["x6d"]

        # Classifier-free conditional guidance
        if (self.cfg_prob is not None) and (self.cfg_prob > 0) and (self.training):
            mask = torch.rand(B, device=x_in.device) < self.cfg_prob
            x_in[mask, 0, ...] = 0

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

        self.ddpm = DenoiseDiffusion(n_steps=1000)
        self.diffusion_cond_keys = diffusion_cond_keys
        self.diffusion_cond_dim = diffusion_cond_dim
        self.cfg_prob = cfg_prob
        self.null_token = None

        self.unet = UNetModel(
            in_channels=1,
            out_channels=1,
            channels=z_dim,
            attention_levels=[0, 1],
            n_res_blocks=1,
            channel_multipliers=[1, 2, 2],
            n_heads=4,
            tf_layers=1,
            d_cond=z_dim + diffusion_cond_dim,
        )

    def get_null(self):
        device = next(self.encoder.parameters()).device
        null_past = torch.zeros((1, self.diffusion_cond_dim))
        null_past = self.encoder(
            torch.zeros((1, self.in_channels, self.window), device=device)
        )[0]
        other_null = torch.zeros((1, self.diffusion_cond_dim), device=device)
        null_token = torch.cat([null_past, other_null], dim=-1)

        return null_token

    def get_samples(self, data, n_steps_ahead, n_samples, uncond_scale, skip_steps):
        batch_size = data["x6d"].shape[0]

        if self.null_token is None:
            self.null_token = self.get_null()

        past_data = {k: v[:, 0] for k, v in data.items() if k in ["x6d", "root"]}
        cond_var = []
        for k in self.diffusion_cond_keys:
            cond_temp = (
                F.one_hot(data[k].ravel().long(), len(self.discrete_classes[k]))
                if k in self.discrete_classes.keys()
                else data[k]
            )
            cond_var += [cond_temp]

        cond_var = torch.cat(cond_var, dim=-1).repeat((n_samples, 1))
        norm_root = self.normalize_root(past_data["root"])

        x_in = torch.cat(
            (past_data["x6d"].view(past_data["x6d"].shape[:-2] + (-1,)), norm_root),
            axis=-1,
        )
        uncond = self.null_token.repeat(batch_size * n_samples, 1)
        init_mu = self.encoder(x_in.moveaxis(-2, -1))[0].reshape(
            batch_size, 1, 1, self.z_dim
        )
        data_o = {"mu": [init_mu.repeat(1, n_samples, 1, 1)]}
        for t_step in range(1, n_steps_ahead + 1):
            cond = torch.cat(
                [data_o["mu"][-1].reshape(-1, self.z_dim), cond_var], dim=-1
            )
            xt = torch.randn_like(data_o["mu"][-1].reshape(-1, self.z_dim))
            time_steps = torch.flip(
                torch.arange(
                    self.ddpm.n_steps, dtype=int, device=past_data["x6d"].device
                ),
                dims=(0,),
            )[skip_steps:]
            for _, step in enumerate(time_steps):
                # import pdb; pdb.set_trace()
                t = xt.new_full((batch_size * n_samples,), step, dtype=torch.long)
                try:
                    xt = self.ddpm.p_sample(
                        self.unet, xt, t, cond, uncond=uncond, uncond_scale=uncond_scale
                    ).squeeze()  ##TODO: Debug multiple samples
                except:
                    import pdb

                    pdb.set_trace()

            data_o["mu"] += [xt.reshape(batch_size, n_samples, 1, self.z_dim)]

        # import pdb; pdb.set_trace()
        data_o["mu"] = torch.cat(data_o["mu"], dim=-2)
        data_o["mu"] = data_o["mu"].reshape(-1, self.z_dim)

        data_o.update(self.decode(data_o["mu"], None))

        data_o = {
            k: v.reshape((batch_size, n_samples, n_steps_ahead + 1) + v.shape[1:])
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

        # Classifier-free conditional guidance
        if (self.cfg_prob is not None) and (self.cfg_prob > 0) and (self.training):
            mask = torch.rand(B, device=x_in.device) < self.cfg_prob
            x_in[mask, 0, ...] = 0

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
        self.diffusion_cond_keys = diffusion_cond_keys
        self.diffusion_cond_dim = diffusion_cond_dim
        self.cfg_prob = cfg_prob

        self.ddpm = DenoiseDiffusion(n_steps=1000)

        self.unet = UNetModel(
            in_channels=1,
            out_channels=1,
            channels=z_dim,
            attention_levels=[0, 1],
            n_res_blocks=1,
            channel_multipliers=[1, 2, 2],
            n_heads=4,
            tf_layers=1,
            d_cond=z_dim + diffusion_cond_dim,
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
        )

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

        x_future = (
            x_in[:, 1, ...].moveaxis(-2, -1).view(-1, self.in_channels, self.window)
        )
        data_o = {}
        data_o[self.dist_params[0]], data_o[self.dist_params[1]] = self.encoder(
            x_future
        )
        data_o["z_past"] = self.past_encoder(x_past)

        if self.prior == "beta":
            # Renormalizing to be between (-1, 1)
            data_o["mu"] = (data_o["alpha"] - 1 + 1e-8) / (
                data_o["alpha"] + data_o["beta"] - 2 + 2e-8
            ) * 2 - 1

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
