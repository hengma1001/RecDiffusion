import os
import glob
import sys
import shutil
import MDAnalysis as mda

sys.path.append("../")


from RecDiffusion.preprocessing import pdbs_to_dbs

n_atoms_threshold = 5000

comp_paths = glob.glob("../data/refined-set/[0-9]*")
# labels = pdbs_to_dbs(comp_paths)

for comp_path in comp_paths:
    label = os.path.basename(comp_path)
    prot_pdb = f"{comp_path}/{label}_protein.pdb"
    lig_mol = f"{comp_path}/{label}_ligand.mol2"

    assert os.path.exists(prot_pdb), f"Missing protein {prot_pdb}"
    assert os.path.exists(lig_mol), f"Missing ligand {lig_mol}"

    prot_u = mda.Universe(prot_pdb)
    prot_u.atoms.names = [name.lstrip("0123456789") for name in prot_u.atoms.names]
    protein_noH = prot_u.select_atoms("protein and not name H*")

    if protein_noH.n_atoms > n_atoms_threshold:
        shutil.move(
            comp_path, f"{os.path.dirname(comp_path)}/_{os.path.basename(comp_path)}"
        )
        print(comp_path, protein_noH.n_atoms)
