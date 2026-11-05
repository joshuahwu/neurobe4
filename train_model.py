import torch

import scrubvae
from scrubvae.params import read
from pathlib import Path
import wandb
import argparse
import warnings
from neurobe4.get import get
from neurobe4.train.trainer import train
from neurobe4.params import PARAM_KEYS

# argparse project and job names
parser = argparse.ArgumentParser(prog="SC-VAE Train", description="Train SC-VAE models")
parser.add_argument("--out_path", "-o", type=str, dest="out_path")
parser.add_argument("--job_id", type=int, dest="job_id")
parser.add_argument("--project", "-p", type=str, dest="project")
parser.add_argument("--name", "-n", type=str, dest="name")
args = parser.parse_args()
if args.out_path is None:
    args.out_path = "/mnt/home/jwu10/working/ceph/results/forecasting/"
### Set/Load Parameters
wandb.login()
if args.job_id is not None:
    z_path = Path(args.out_path + args.project)
    folders = sorted([str(f.parts[-1]) for f in z_path.iterdir() if f.is_dir()])
    name = folders[args.job_id]
    # analysis_key = "{}/{}/".format(args.project, name)
else:
    name = args.name

# Read in config file with all parameters and settings
config = read.config(
    "{}/{}/{}/model_config.yaml".format(args.out_path, args.project, name), 
    param_keys=PARAM_KEYS
)

condition_tau = "tau" in (config["model"]["conditional_keys"] or [])
if config["data"]["random_tau"] and not condition_tau:
    warnings.warn(
        "data.random_tau is true but model.conditional_keys does not include "
        "'tau': the model will be trained on mixed forecast horizons without "
        "being told each sample's horizon"
    )
elif condition_tau and not config["data"]["random_tau"]:
    warnings.warn(
        "model.conditional_keys includes 'tau' but data.random_tau is not true: "
        "batches have no tau, so every sample gets the null tau token and the "
        "model never sees a real horizon"
    )

# Initialize Weights & Biases
run = wandb.init(
    project=args.project, name=name, config=config, dir=args.out_path + args.project + "/" + name
)
print("WANDB directory: {}".format(run.dir))

device = "cuda" if torch.cuda.is_available() else "cpu"
# Get DataLoaders and model
loader_dict, model = get.get_data_and_model(
    config,
    train_val_test=["train", "val"],
    shuffle=[True,False],
    device=device,
)

# Train model
model = train(config=config, model=model, loader_dict=loader_dict, run=run, device=device)

run.finish()
