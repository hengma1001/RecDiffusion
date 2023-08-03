import torch

import numpy as np
import MDAnalysis as mda



def pdb_to_dict(
        prot_pdb,
        lig_mol2,
        # input_feature=None,
        node_attribute=None) -> dict:
    data = {}

    prot_u = mda.Universe(prot_pdb)
    prot_u.atoms.names = [name.lstrip('0123456789') for name in prot_u.atoms.names]
    protein_noH = prot_u.select_atoms('protein and not name H*')

    lig_u = mda.Universe(lig_mol2)
    lig_noH = lig_u.select_atoms('not name H*')
    
    positions = np.concatenate([protein_noH.positions, lig_noH.positions], axis=0)
    positions = positions - np.mean(positions, axis=0)
    data['pos'] = torch.tensor(positions)
    # if input_feature:
    #     assert len(input_feature) == protein.atoms.n_atoms
    #     data['x'] = input_feature
    if node_attribute:
        data['y'] = [atom.resname + atom.name for atom in protein_noH.atoms] + \
                ['LIG' + elem for elem in lig_noH.atoms.elements]
    return data


def position_to_pdb(prot_pdb, lig_mol2, pdb_output, positions):
    prot_u = mda.Universe(prot_pdb)
    prot_u.atoms.names = [name.lstrip('0123456789') for name in prot_u.atoms.names]
    protein_noH = prot_u.select_atoms('protein and not name H*')

    lig_u = mda.Universe(lig_mol2)
    lig_noH = lig_u.select_atoms('not name H*')

    full_u = mda.Merge(protein_noH.atoms, lig_noH.atoms)
    full_u.atoms.positions = positions
    full_u.atoms.write(pdb_output)


def attr_to_onehot(res_atom_name):
    pass