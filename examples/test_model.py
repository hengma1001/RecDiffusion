import glob
import sys

import e3nn
import torch

sys.path.append("../")

from typing import Any, Dict

from pydantic import BaseModel, root_validator

from RecDiffusion.model import e3_diffusion
from RecDiffusion.preprocessing import pdbs_to_dbs

# class Settings(BaseModel):
#     irreps_in: e3nn.o3.Irreps = e3nn.o3.Irreps("16x0e")
#     number_of_basis: 10

#     @root_validator(pre=True)
#     def custom_type_validator(cls, values: Dict[str, Any]) -> Dict[str, Any]:
#         if "irreps_in" in values:
#             values["irreps_in"] = e3nn.o3.Irreps(values["irreps_in"])
#         return values


time_step = 500
scheduler = "linear_beta_schedule"

comp_paths = glob.glob("../data/refined-set/2w*")
dbs, full_voca_size, labelencoder = pdbs_to_dbs(
    comp_paths, prot_sel="protein and not name H*", node_attr=True
)

model_kwargs = {
    "irreps_in": "16x0e",  # no input features
    "irreps_hidden": "32x0e + 32x0o + 32x1e + 32x1o",  # hyperparameter
    "irreps_out": "1x1o",  # 12 vectors out, but only 1 vector out per input
    "irreps_node_attr": "10x0e",
    "irreps_edge_attr": 3,
    "layers": 3,  # hyperparameter
    "max_radius": 3.5,
    "number_of_basis": 10,
    "radial_layers": 1,
    "radial_neurons": 128,
    "num_neighbors": 11,  # average number of neighbors w/in max_radius
    "num_nodes": 12,  # not important unless reduce_output is True
    "node_attr_n_kind": full_voca_size,
    "node_attr_emb_dim": 10,
    "time_emb_dim": 16,
    "radius_decay": 400,
    "reduce_output": False,  # setting this to true would give us one scalar as an output.
}

device = "cuda"
model = e3_diffusion(time_step, scheduler, **model_kwargs).to(device)

time = torch.Tensor([0]).to(device)
with torch.no_grad():
    print(model(dbs[0].to(device), torch.Tensor([0]).to(device)))
