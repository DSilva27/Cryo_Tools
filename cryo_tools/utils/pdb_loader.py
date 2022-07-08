import MDAnalysis as mda

def load_pdb(pdb_fname, filter):

    atomic_model = mda.Universe(pdb_fname)
    atomic_model.atoms.translate(-atomic_model.select_atoms(filter).center_of_mass())

    coordinates = atomic_model.select_atoms(filter).positions.T

    return coordinates