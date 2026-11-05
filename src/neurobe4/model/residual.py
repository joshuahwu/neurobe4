import torch
import torch.nn as nn
import torch.nn.functional as F
from scrubvae.model.residual import CholeskyL, ResidualBlock, find_latent_dim


class ResidualEncoder(nn.Module):
    """
    Copied from scrubvae.model.residual.ResidualEncoder, with an option to
    skip the flatten + linear bottleneck and return a sequence of tokens.
    With bottleneck=True it is identical to scrubvae's (same parameter names,
    so scrubvae checkpoints load into it).
    """

    def __init__(
        self,
        in_channels,
        ch=[64, 128, 256, 512, 1024],
        kernel=5,
        z_dim=128,
        window=200,
        activation="prelu",
        is_diag=False,
        prior="gaussian",
        init_dilation=None,
        bottleneck=True,
    ):
        """
        bottleneck : bool
            If True, flatten + linear layer(s) to z_dim (or distribution
            parameters), as in scrubvae. If False (requires prior=None),
            forego the bottleneck: forward returns one token per time step of
            the final convolutional feature map, (N, n_tokens, z_dim), each
            projected from ch[-1] channels to z_dim plus a learned positional
            embedding.
        """
        super(ResidualEncoder, self).__init__()
        self.prior = prior
        self.bottleneck = bottleneck
        self.conv_in = nn.Conv1d(in_channels, ch[0], 7, 1, 3)
        self.activation = nn.Tanh() if activation == "tanh" else nn.PReLU()

        if init_dilation is None:
            dilation = torch.ones(len(ch) - 1, dtype=int)
        else:
            dilation = init_dilation * 2 ** torch.arange(len(ch) - 1)

        layers = []
        for i in range(len(ch) - 1):
            layers += [
                ResidualBlock(ch[i], ch[i + 1], kernel, activation, dilation[i].item())
            ]
        self.res_layers = nn.Sequential(*layers)

        n_latent = find_latent_dim(window, kernel, len(ch) - 1, dilation)

        if not bottleneck:
            assert prior is None, "bottleneck=False requires prior=None"
            self.n_tokens = n_latent
            self.token_proj = nn.Linear(ch[-1], z_dim)
            self.token_pos = nn.Parameter(0.02 * torch.randn(1, n_latent, z_dim))
            return

        self.flatten = nn.Flatten()

        flatten_dim = n_latent * ch[-1]

        if prior == "gaussian":
            sig_dim = z_dim if is_diag else z_dim * (z_dim + 1) // 2
            self.fc_mu = nn.Linear(flatten_dim, z_dim)
            self.fc_sigma = nn.Sequential(
                nn.Linear(flatten_dim, sig_dim), CholeskyL(z_dim, is_diag)
            )
        elif prior == "beta":
            self.fc_alpha = nn.Linear(flatten_dim, z_dim)
            self.fc_beta = nn.Linear(flatten_dim, z_dim)
        elif prior is None:
            self.fc = nn.Linear(flatten_dim, z_dim)

    def forward(self, x):
        x = self.activation(self.conv_in(x))
        x = self.res_layers(x)
        
        if not self.bottleneck:
            # (N, ch[-1], n_tokens) -> (N, n_tokens, z_dim)
            return self.token_proj(x.transpose(1, 2)) + self.token_pos
        
        x = self.flatten(x)
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
