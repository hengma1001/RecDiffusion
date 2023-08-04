import torch
import e3nn

import sys

sys.path.append("../")

from RecDiffusion.model import Network
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


time = 100
model_kwargs = {
    "irreps_in": e3nn.o3.Irreps("16x0e"),  # no input features
    "irreps_hidden": e3nn.o3.Irreps("32x0e + 32x0o + 32x1e + 32x1o"),  # hyperparameter
    "irreps_out": e3nn.o3.Irreps(
        "1x1o"
    ),  # 12 vectors out, but only 1 vector out per input
    "irreps_node_attr": e3nn.o3.Irreps("19x0e"),
    "irreps_edge_attr": e3nn.o3.Irreps.spherical_harmonics(3),
    "layers": 3,  # hyperparameter
    "max_radius": 3.5,
    "number_of_basis": 10,
    "radial_layers": 1,
    "radial_neurons": 128,
    "num_neighbors": 11,  # average number of neighbors w/in max_radius
    "num_nodes": 12,  # not important unless reduce_output is True
    "time_emb_dim": 16,
    "reduce_output": False,  # setting this to true would give us one scalar as an output.
}

model = Network(**model_kwargs).cuda()

data = {}
data["pos"] = torch.rand([29, 3]).cuda()
# data['z'] = torch.randint(1, 100, (29, 1))
data["z"] = torch.rand([29, 19]).cuda()
time = torch.randint(300, (1,)).cuda()

print(time)
print(model(data, time))
