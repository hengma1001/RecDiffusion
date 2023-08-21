import sys

import e3nn
import MDAnalysis as mda
import torch

sys.path.append("../")

from RecDiffusion.diffusion import diffusion_sampler
from RecDiffusion.model import Network
from RecDiffusion.preprocessing import pdb_to_dict, position_to_pdb

# model_kwargs = {
#     "irreps_in": "16x0e",  # no input features
#     "irreps_hidden": "32x0e + 32x0o + 32x1e + 32x1o",  # hyperparameter
#     "irreps_out": "1x1o",  # 12 vectors out, but only 1 vector out per input
#     "irreps_node_attr": "10x0e",
#     "irreps_edge_attr": 3,
#     "layers": 3,  # hyperparameter
#     "max_radius": 3.5,
#     "number_of_basis": 10,
#     "radial_layers": 1,
#     "radial_neurons": 128,
#     "num_neighbors": 11,  # average number of neighbors w/in max_radius
#     "num_nodes": 12,  # not important unless reduce_output is True
#     "node_attr_n_kind": full_voca_size,
#     "node_attr_emb_dim": 10,
#     "time_emb_dim": 16,
#     "reduce_output": False,  # setting this to true would give us one scalar as an output.
# }

# model = Network(**model_kwargs)

timesteps = 500
prot_pdb = "../data/refined-set/2w47/2w47_protein.pdb"
lig_mol2 = "../data/refined-set/2w47/2w47_ligand.mol2"

noise_lvls = [10, 20, 40, 100, 150, 200, 250, 300, 400, 500]
data = pdb_to_dict(prot_pdb, lig_mol2, node_attr=True)

sampling = diffusion_sampler(timesteps)
for t in noise_lvls:
    save_pdb = f"test_{t:03}.pdb"
    positions = sampling.q_sample(data["pos"], t - 1)
    position_to_pdb(prot_pdb, lig_mol2, save_pdb, positions)

# a = sampling.sample_pose(model, data)
# print(a.shape)
