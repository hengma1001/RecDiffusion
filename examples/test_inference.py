import glob
import sys
from functools import partial

import torch
import torch_geometric

sys.path.append("../")

from RecDiffusion.model import e3_diffusion
from RecDiffusion.preprocessing import path_to_pdb, position_to_pdb
from RecDiffusion.utils import dict_from_yaml

input_dict = "./rec_diffusion/input.yml"
input_dict = dict_from_yaml(input_dict)

model_checkpnt = "/lambda_stor/homes/heng.ma/Research/md_pkgs/RecDiffusion/examples/rec_diffusion/g8lafriy/checkpoints/epoch=49-step=110850.ckpt"

model = e3_diffusion.load_from_checkpoint(model_checkpnt)

test_data = torch.load(input_dict["test"])
data = test_data.dataset[0]
comp_path = f"../data/refined-set/{data.sys_name}"

# model.diffu_sampler.p_sample
x_noisy = model.diffu_sampler.q_sample(data["pos"], model.time_step - 1)

data_noisy = torch_geometric.data.Data(
    pos=x_noisy,
    z=data.z,
)


prot_pdb, lig_mol = path_to_pdb(comp_path)
pose_to_pdb = partial(position_to_pdb, prot_pdb, lig_mol, "test.pdb")

pose_denoised = model.diffu_sampler.sample_pose(model, data_noisy)
