import torch
import e3nn
import glob
import sys
import torch
from torch.profiler import profile, record_function, ProfilerActivity

sys.path.append("../")

from RecDiffusion.model import e3_diffusion
from RecDiffusion.preprocessing import pdbs_to_datasets, pdbs_to_dbs
from RecDiffusion.training import train_diffu
from typing import Dict, Any
from pydantic import BaseModel, root_validator


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
train, val, test, full_voca_size = pdbs_to_datasets(comp_paths)

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
    "reduce_output": False,  # setting this to true would give us one scalar as an output.
}

model = e3_diffusion(time_step, scheduler, **model_kwargs).cuda()


# comp_paths = glob.glob("../data/refined-set/2w*")
# labels = pdbs_to_dbs(comp_paths)

# time = torch.randint(300, (1,)).cuda()
# print(time)
# print(model(labels[0].cuda(), time))

model, result = train_diffu(model, train, val, test, n_gpus=1)
# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     profile_memory=True,
#     record_shapes=True,
# ) as prof:
#     with record_function("model_inference"):
#         model._get_loss(labels[0].cuda())
