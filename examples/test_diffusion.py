import torch
import e3nn
import glob
import sys

sys.path.append("../")

from RecDiffusion.model import e3_diffusion
from RecDiffusion.preprocessing import pdbs_to_dbs
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

prot_pdb = "../data/refined-set/2wed/2wed_protein.pdb"
lig_mol2 = "../data/refined-set/2wed/2wed_ligand.mol2"

comp_paths = glob.glob("../data/refined-set/2w*")
labels, full_voca_size = pdbs_to_dbs(comp_paths)

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

device = "cuda"
model = e3_diffusion(time_step, scheduler, **model_kwargs).to(device)


eta = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=eta)
optimizer.zero_grad()

# time = torch.randint(300, (1,)).to(device)
# print(time)
# with torch.no_grad():
#     print(model(labels[0].to(device), time))


def train_step(label):
    loss = model._get_loss(label.to(device))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


loss = model._get_loss(labels[0].to(device))
print(loss)
