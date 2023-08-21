import glob
import sys

import torch
import wandb
from pytorch_lightning.loggers import WandbLogger

sys.path.append("../")

from RecDiffusion.model import e3_diffusion
from RecDiffusion.training import train_diffu
from RecDiffusion.utils import dict_from_yaml

wandb_logger = WandbLogger(
    project="rec_diffusion", log_model="all"  # group runs in "MNIST" project
)

time_step = 500
scheduler = "linear_beta_schedule"
time_emb_dim = 16
node_attr_emb_dim = 10
radius_decay = time_step * 0.8

input_dict = "./rec_diffusion/input.yml"
input_dict = dict_from_yaml(input_dict)

full_voca_size = input_dict["full_voca_size"]

model_kwargs = {
    "irreps_in": f"{time_emb_dim}x0e",  # no input features
    "irreps_hidden": "32x0e + 32x0o + 32x1e + 32x1o",  # hyperparameter
    "irreps_out": "1x1o",  # 12 vectors out, but only 1 vector out per input
    "irreps_node_attr": f"{node_attr_emb_dim}x0e",
    "irreps_edge_attr": 3,
    "layers": 3,  # hyperparameter
    "max_radius": 3.5,
    "number_of_basis": 10,
    "radial_layers": 1,
    "radial_neurons": 128,
    "num_neighbors": 11,  # average number of neighbors w/in max_radius
    "num_nodes": 12,  # not important unless reduce_output is True
    "node_attr_n_kind": full_voca_size,
    "node_attr_emb_dim": node_attr_emb_dim,
    "time_emb_dim": time_emb_dim,
    "radius_decay": radius_decay,
    "reduce_output": False,  # setting this to true would give us one scalar as an output.
}

model = e3_diffusion(time_step, scheduler, **model_kwargs)

for db_name in ["train", "val", "test"]:
    exec(f"{db_name} = torch.load('{input_dict[db_name]}')")

model, result = train_diffu(
    model,
    train,
    val,
    test,
    n_gpus=1,
    max_epochs=100,
    every_n_epochs=10,
    logger=wandb_logger,
)
