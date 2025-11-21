import torch.nn as nn
import torch.nn.functional as F
import torch
import scrubvae
from neuroposelib import read, preprocess
import numpy as np

CODE_PATH = "/mnt/home/jwu10/working/ssumo/"
RESULTS_PATH = "/mnt/ceph/users/jwu10/results/vae/"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        # import pdb; pdb.set_trace()
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class GeGLU(nn.Module):
    """
    ### GeGLU Activation

    $$\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$$
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # Combined linear projections $xW + b$ and $xV + c$
        self.proj = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor):
        # Get $xW + b$ and $xV + c$
        x, gate = self.proj(x).chunk(2, dim=-1)
        # $\text{GeGLU}(x) = (xW + b) * \text{GELU}(xV + c)$
        return x * F.gelu(gate)


class BasicTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=0)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            GeGLU(embed_dim, embed_dim * 4), nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, cross=None):
        x_norm = self.norm1(x)
        x = self.attn1(x_norm, x_norm, x_norm) + x
        x_norm = self.norm2(x)
        if cross == None:
            cross = x_norm
        x = self.attn2(x_norm, cross, cross) + x
        x_norm = self.norm3(x)
        x = self.ff(x_norm, x_norm, x_norm) + x
        return x


class TransformerForecast(nn.Module):
    def __init__(
        self,
        z_dim,
        activation="gelu",
        n_heads=4,
        ff_size=512,
        n_layers=4,
        channel_multiplier=2,
        seq_len=5,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.pos_encoder = PositionalEncoding(z_dim, dropout=0.1)

        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        for i in range(n_layers):
            self.encoder.append([BasicTransformerBlock(z_dim, n_heads)])

        self.decoder = nn.ModuleList([])
        for i in range(n_layers):
            self.decoder.append([BasicTransformerBlock(z_dim, n_heads)])

        self.fc_mu = nn.Linear(z_dim, z_dim)
        self.fc_logvar = nn.Linear(z_dim, z_dim)

    def sampling(self, mu, logvar):
        """Reparameterization trick

        Parameters
        ----------
        mu : torch.tensor
            Batch_size x latent dimensions
        L : torch.tensor
            Batch_size x latent dimensions.
        """
        eps = torch.randn_like(mu)
        return (logvar[None,...]*eps[..., None]).squeeze().add_(mu)
    
    def forward(self, z):
        z = self.pos_encoder(z)
        z_encode = [self.encoder[0](z)]
        for i in range(1, self.n_layers):
            z_encode.append(self.encoder[i](z_encode[-1]))

        z_decode = self.decoder[0](z_encode.pop())
        for i in range(1, self.n_layers):
            z_decode = self.decoder[i](z_decode, cross=z_encode.pop())
        return z
        

config = read.config(RESULTS_PATH + "vanilla/64/model_config.yaml")
connectivity = read.connectivity_config(config["data"]["skeleton_path"])
epoch = 300
stride = 1
n_steps = 4

loader = scrubvae.get.mouse_data(
    data_config=config["data"],
    window=config["model"]["window"],
    train=True,
    data_keys=["x6d", "root", "offsets", "avg_speed"],
    shuffle=False,
    normalize=[],
)
window = config["model"]["window"]
n_keypts = loader.dataset.n_keypts
kinematic_tree = loader.dataset.kinematic_tree

z = scrubvae.get.latents(
    config,
    model=None,
    epoch=epoch,
    loader=loader,
    dataset_label="Train",
)

window_inds = loader.dataset.window_inds
t_diffs = np.diff(window_inds[:, window // 2], prepend=[-10], append=[-10])
cont_ids = np.diff(np.where(t_diffs != 1)[0])
cont_ids = np.concatenate([np.zeros(i) + j for j, i in enumerate(cont_ids)])

z_inds = scrubvae.data.dataset.get_window_indices(cont_ids, stride, n_steps + 1)

T4 = TransformerForecast(z.shape[-1], seq_len=n_steps - 1).cuda()
T4(z[z_inds[0:2]].cuda())


import pdb

pdb.set_trace()
