import MDAnalysis as mda

def pdb_to_dict(pdb_file, input_feature=None, node_attribute=None):
    data = {}
    mda_u = mda.Universe(pdb_file)
    data['pos'] = mda_u.atoms.positions
    if input_feature:
        assert len(input_feature) == mda_u.atoms.n_atoms
        data['x'] = input_feature
    if node_attribute:
        data['y'] = mda_u.atoms.names
    return data