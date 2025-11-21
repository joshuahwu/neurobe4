import torch
import torch.optim as optim
import tqdm
import time
import wandb
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
from line_profiler import profile
from pathlib import Path
from scrubvae.model.residual import ConditionalForecaster, ForecastingVAE
from neurobe4.model.model import (
    DenoiseDiffusion,
    DiffVAE,
)


class CyclicalBetaAnnealing(torch.nn.Module):
    def __init__(self, beta_max=1, len_cycle=100, R=0.5):
        self.beta_max = beta_max
        self.len_cycle = len_cycle
        self.R = R
        self.len_increasing = int(len_cycle * R)

    def get(self, epoch):
        remainder = (epoch - 1) % self.len_cycle
        if remainder >= self.len_increasing:
            beta = self.beta_max
        else:
            beta = self.beta_max * remainder / self.len_increasing

        return beta

def get_beta_schedule(schedule, beta):
    if schedule == "cyclical":
        print("Initializing cyclical beta annealing")
        beta_scheduler = CyclicalBetaAnnealing(beta_max=beta)
    else:
        print("No beta annealing selected")
        beta_scheduler = None

    return beta_scheduler


def get_optimizer_and_lr_scheduler(
    model, train_config, load_path=None, start_epoch=None
):
    """
    Loads in optimizer and learning rate schedulers
    """
    if train_config["optimizer"] == "adam":
        print("Initializing Adam optimizer ...")
        optimizer = optim.Adam(model.parameters(), lr=train_config["lr"])
    elif train_config["optimizer"] == "adamw":
        print("Initializing AdamW optimizer ...")
        optimizer = optim.AdamW(model.parameters(), lr=train_config["lr"])
    elif train_config["optimizer"] == "sgd":
        print("Initializing SGD optimizer ...")
        optimizer = optim.SGD(
            model.parameters(), lr=train_config["lr"], momentum=0.2, nesterov=True
        )
    else:
        raise ValueError("No valid optimizer selected")

    if train_config["lr_schedule"] == "cawr":
        print("Initializing cosine annealing w/warm restarts learning rate scheduler")
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
    elif train_config["lr_schedule"] is None:
        print("No learning rate scheduler selected")
        scheduler = None

    if load_path is not None:
        if Path("{}/checkpoints/epoch_{}.pth".format(load_path, start_epoch)).exists():
            checkpoint = torch.load(
                "{}/checkpoints/epoch_{}.pth".format(load_path, start_epoch)
            )
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except:
                print("Optimizer state dict unable to be loaded")

            scheduler = checkpoint["lr_scheduler"]

    return optimizer, scheduler


def predict_batch(model, data):
    data_i = {k: v for k, v in data.items() if (k in ["x6d", "root", "var"])}

    return model(data_i)


@profile
def train_test_epoch(
    config,
    model,
    loader,
    device,
    epoch,
    optimizer=None,
    scheduler=None,
    mode="train",
    # get_z=False,
):
    model.train()
    if config["train"]["freeze_ae"]:
        # if isinstance(model, ConditionalDiffusion):
        #     for param in model.future_encoder.parameters():
        #         param.requires_grad = False
        # else:
        for param in model.encoder.parameters():
            param.requires_grad = False

        for param in model.decoder.parameters():
            param.requires_grad = False

    model.mi_estimator = None
    epoch_metrics = {k: 0 for k in ["total"] + list(config["loss"].keys())}
    for batch_idx, data in enumerate(loader):
        data = {k: v.to(device) for k, v in data.items()}

        data_o = predict_batch(model, data)

        batch_loss = get_batch_loss(
            model,
            data,
            data_o,
            config["loss"],
            model.ddpm if hasattr(model, "ddpm") else None,
        )

        if mode == "train":
            # Update model parameters
            for param in model.parameters():
                param.grad = None
            
            batch_loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6)
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch + batch_idx / len(loader))

        # Save loss and other metrics for the epoch
        epoch_metrics = {
            k: v + batch_loss[k].detach() for k, v in epoch_metrics.items()
        }

    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item() / len(loader)
        print(
            "====> Epoch: {} Average {} loss: {:.4f}".format(epoch, k, epoch_metrics[k])
        )

    return epoch_metrics


def test_epoch(config, model, loader, device="cuda", epoch=0):
    print("Running test epoch")

    model.eval()
    with torch.no_grad():
        # z = []
        epoch_metrics = {k: 0 for k in ["total"] + list(config["loss"].keys())}

        # if (
        #     # isinstance(model, ConditionalDiffusion)
        #     isinstance(model, NCConditionalDiffusion)
        #     or isinstance(model, DiffVAE)
        # ):
        #     ddpm = DenoiseDiffusion(n_steps=1000)
        # else:
        #     ddpm = None

        for batch_idx, data in enumerate(loader):
            data = {k: v.to(device) for k, v in data.items()}
            data_o = predict_batch(model, data)

            batch_metrics = get_batch_loss(
                model,
                data,
                data_o,
                config["loss"],
                model.ddpm if hasattr(model, "ddpm") else None,
            )
            epoch_metrics = {
                k: v + batch_metrics[k].detach() for k, v in epoch_metrics.items()
            }

    for k, v in epoch_metrics.items():
        epoch_metrics[k] = v.item() / len(loader)
        print(
            "====> Epoch: {} Average {} loss: {:.4f}".format(epoch, k, epoch_metrics[k])
        )

    return epoch_metrics  # , torch.cat(z, dim=0).cpu()


def train_epoch(config, model, loader, optimizer, scheduler, device="cuda", epoch=0):
    epoch_metrics = train_test_epoch(
        config=config,
        model=model,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epoch=epoch,
        mode="train",
        # get_z=False,
    )

    return epoch_metrics

@profile
def train(config, model, loader_dict, run=None):
    torch.set_float32_matmul_precision("medium")
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True

    optimizer, scheduler = get_optimizer_and_lr_scheduler(
        model,
        config["train"],
        config["model"]["load_model"],
        config["model"]["start_epoch"],
    )

    # Apply beta annealing if selected
    if "prior" in config["loss"].keys():
        beta_scheduler = get_beta_schedule(
            config["loss"]["prior"],
            config["train"]["beta_anneal"],
        )
    else:
        beta_scheduler = None

    # Epoch loop
    for epoch in tqdm.trange(
        config["model"]["start_epoch"] + 1, config["train"]["num_epochs"] + 1
    ):
        # Update beta annealing if applicable
        if beta_scheduler is not None:
            config["loss"]["prior"] = beta_scheduler.get(epoch)
            print("Beta schedule: {:.3f}".format(config["loss"]["prior"]))

        starttime = time.time()
        # Train for an epoch
        train_metrics = train_epoch(
            config=config,
            model=model,
            loader=loader_dict["train"],
            optimizer=optimizer,
            scheduler=scheduler,
            device="cuda",
            epoch=epoch,
        )
        metrics = {"{}_train".format(k): v for k, v in train_metrics.items()}

        metrics["time"] = time.time() - starttime

        if epoch % 5 == 0:
            print("Saving model to folder: {}".format(config["out_path"]))
            torch.save(
                {k: v.cpu() for k, v in model.state_dict().items()},
                "{}/weights/epoch_{}.pth".format(config["out_path"], epoch),
            )

            if epoch % 20 == 0:
                torch.save(
                    {"optimizer": optimizer.state_dict(), "lr_scheduler": scheduler},
                    "{}/checkpoints/epoch_{}.pth".format(config["out_path"], epoch),
                )

            ## Calculate test metrics
            if epoch >= 50:
                test_metrics = test_epoch(
                    config=config,
                    model=model,
                    loader=loader_dict["val"],
                    device="cuda",
                    epoch=epoch,
                )
                metrics.update(
                    {"{}_test".format(k): v for k, v in test_metrics.items()}
                )

        wandb.log(metrics, epoch)

    return model
