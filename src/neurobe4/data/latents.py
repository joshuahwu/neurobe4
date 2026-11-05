import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from neurobe4.data.dataset import RodentDiskDataset, collate_h5_dataset
from neurobe4.data.train_utils import prepare_batch


class _WindowDataset(Dataset):
    """Every window of a RodentDiskDataset split (dataset.centers of each h5
    file in turn, the order of per-window arrays), as single-window poses"""

    def __init__(self, dataset: RodentDiskDataset):
        self.dataset = dataset
        self.windows = [
            (h5_key, int(c))
            for h5_key, centers in dataset.centers.items()
            for c in centers
        ]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        h5_key, center = self.windows[idx]
        start = center - self.dataset.model_window // 2
        pose = self.dataset.h5_datasets[h5_key][start : start + self.dataset.model_window]
        return {"pose": torch.from_numpy(pose).to(self.dataset.pose_dtype)}


def get_cached_latents(
    dataset: RodentDiskDataset,
    vae,
    cache_path: str,
    batch_size: int = 1024,
    num_workers: int = 5,
    device: str = "cuda",
) -> np.ndarray:
    """
    Latent mean of every window of `dataset`'s split, encoded by `vae`. Loaded
    from `cache_path` if it exists, otherwise computed and saved there first.

    Each window is encoded once, whatever the stride or time-pairing; samples
    look up their windows' rows with dataset.window_rows.

    Parameters
    ----------
    dataset : RodentDiskDataset or RodentDiskDatasetTPaired
    vae
        Model with encode({"x6d", "root"}) -> {"mu": (N, z_dim)} taking single
        windows (e.g. scrubvae's ResVAE).
    cache_path : str
        .npy file. Should uniquely identify the split's windows (see
        dataset.cache_name()) and the vae used.

    Returns
    -------
    np.ndarray (read-only memmap), (dataset.n_windows, z_dim)
    """
    if Path(cache_path).is_file():
        print("Loading cached latents from {}".format(cache_path))
        latents = np.load(cache_path, mmap_mode="r")
        if len(latents) != dataset.n_windows:
            raise ValueError(
                "Cached latents at {} have {} windows but the dataset has {}; "
                "delete the file to recompute".format(
                    cache_path, len(latents), dataset.n_windows
                )
            )
        return latents

    print(
        "Encoding latents for {} windows -> {}".format(dataset.n_windows, cache_path)
    )
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        _WindowDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_h5_dataset,
    )
    # Written to a per-process temporary file then renamed, so an interrupted
    # run never leaves a partial cache behind and concurrent jobs (e.g. array
    # tasks encoding the same split) never write or delete each other's file
    tmp_path = str(cache_path)[: -len(".npy")] + f".tmp{os.getpid()}.npy"
    latents = None
    i = 0
    vae.eval()
    # Full fp32 so cached values don't depend on batch size/composition
    # (TF32 convolutions, on by default in cuDNN, can shift them by ~1e-2)
    tf32 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.no_grad():
            for data in tqdm.tqdm(loader):
                data = prepare_batch(data, dataset, device=device)
                mu = vae.encode({"x6d": data["x6d"], "root": data["root"]})["mu"]
                mu = mu.cpu().numpy()

                if latents is None:
                    latents = np.lib.format.open_memmap(
                        tmp_path,
                        mode="w+",
                        dtype=np.float32,
                        shape=(dataset.n_windows, mu.shape[-1]),
                    )
                latents[i : i + len(mu)] = mu
                i += len(mu)
        latents.flush()
        del latents
        os.replace(tmp_path, cache_path)
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
        if Path(tmp_path).is_file():
            os.remove(tmp_path)

    return np.load(cache_path, mmap_mode="r")


def get_latent_norm(
    dataset,
    latents: np.ndarray,
    forecasting_type: str,
    latent_norm: str,
    max_samples: int = 2**20,
    chunk: int = 2**18,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Diffusion target normalization statistics over a time-paired dataset's
    samples, from per-window latents (see get_cached_latents).

    - "velocity": the target mu_tau - mu_0 grows with the forecast horizon tau,
      so statistics are computed per tau: rows tau = 0..n_future_windows
      (row 0, unused, is the identity), indexed by each sample's tau.
    - "center": the target mu_tau is a future latent whose distribution does
      not depend on tau, so one shared row (computed at tau = 1).

    Each horizon uses up to max_samples start windows (a random subset if
    there are more), accumulated in chunks so memmapped latents never need to
    fit in memory.

    Returns
    -------
    {"mu": (n_rows, z_dim), "std": (n_rows, z_dim)} float32 tensors
    ("scalar" mode: mu = 0 and one std per row shared by all dimensions)
    """
    if forecasting_type not in ("velocity", "center"):
        raise ValueError(f"Unknown forecasting_type {forecasting_type}")
    if latent_norm not in ("norm", "scalar"):
        raise ValueError(f"Unknown latent_norm {latent_norm}")

    # Start windows to use: (h5_key, starts) pairs, subsampled if needed
    n_total = sum(len(s) for s in dataset.indices.values())
    keep = None
    if n_total > max_samples:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(n_total, max_samples, replace=False))
    starts_by_key, offset = [], 0
    for h5_key, starts in dataset.indices.items():
        if keep is None:
            selected = starts
        else:
            local = keep[(keep >= offset) & (keep < offset + len(starts))] - offset
            selected = starts[local]
        offset += len(starts)
        starts_by_key.append((h5_key, selected))

    taus = range(1, dataset.n_future_windows + 1) if forecasting_type == "velocity" else [1]
    z_dim = latents.shape[-1]
    n_rows = dataset.n_future_windows + 1 if forecasting_type == "velocity" else 1
    mu_rows = np.zeros((n_rows, z_dim))
    std_rows = np.ones((n_rows, z_dim))
    for tau in taus:
        total, total_sq, n = 0.0, 0.0, 0
        for h5_key, starts in starts_by_key:
            for s in range(0, len(starts), chunk):
                start = starts[s : s + chunk]
                feat = np.asarray(
                    latents[dataset.window_rows(h5_key, start + tau * dataset.n_skip)],
                    dtype=np.float64,
                )
                if forecasting_type == "velocity":
                    feat = feat - np.asarray(
                        latents[dataset.window_rows(h5_key, start)], dtype=np.float64
                    )
                total = total + feat.sum(0)
                total_sq = total_sq + (feat**2).sum(0)
                n += len(feat)

        mu = total / n
        row = tau if forecasting_type == "velocity" else 0
        if latent_norm == "norm":
            mu_rows[row] = mu
            # unbiased, like torch.std
            std_rows[row] = np.sqrt((total_sq / n - mu**2) * n / (n - 1))
        else:
            # one std over all latent dimensions together; no centering
            mu_all = mu.mean()
            var_all = (total_sq.sum() / (n * len(mu)) - mu_all**2) * (
                n * len(mu) / (n * len(mu) - 1)
            )
            std_rows[row] = np.sqrt(var_all)

    return {
        "mu": torch.tensor(mu_rows, dtype=torch.float32),
        "std": torch.tensor(std_rows, dtype=torch.float32),
    }
