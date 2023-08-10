import os
import glob
import sys
import shutil
import MDAnalysis as mda

sys.path.append("../")


from RecDiffusion.preprocessing import pdbs_to_dbs

n_atoms_threshold = 5000

comp_paths = glob.glob("../data/refined-set/_*")
# labels = pdbs_to_dbs(comp_paths)

for comp_path in comp_paths:
    shutil.move(
        comp_path, f"{os.path.dirname(comp_path)}/{os.path.basename(comp_path)[1:]}"
    )
    print(comp_path)
