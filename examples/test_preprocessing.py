import glob
import torch
import e3nn

import MDAnalysis as mda
import sys
sys.path.append('../')

from RecDiffusion.preprocessing import pdb_to_dict, pdbs_to_dbs
from RecDiffusion.diffusion import diffusion_sample

timesteps = 500
prot_pdb = "../data/refined-set/2wed/2wed_protein.pdb"
lig_mol2 = "../data/refined-set/2wed/2wed_ligand.mol2"

noise_lvls = [10, 20, 40, 100, 150, 200, 250, 300, 400, 500]

sampling = diffusion_sample(timesteps)
data = pdb_to_dict(prot_pdb, lig_mol2,
                   node_attr=True)

print(data)

comp_paths = glob.glob("../data/refined-set/2w*")

labels = pdbs_to_dbs(comp_paths)
