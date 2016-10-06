from __future__ import division, print_function
import numpy as np

from rdkit import Chem

import features

def tensorise_smiles(smiles, max_degrees=5, max_atoms=100):
    '''Takes a list of smiles and turns the graphs in tensor representation.

    Args:
        smiles: a list of smiles representations

    Returns:
        list of np.array:
           - atoms: An atom feature np.array of size `(molecules, max_atoms, atom_features)`
           - bonds: A bonds np.array of size `(molecules, max_atoms, max_neighbours)`
           - edges: A connectivity array of size `(molecules, max_atoms, max_neighbours, bond_features)`
    TODO:
        * Auto-option for max_degrees and max_atoms
        * Arguments for sparse vector encoding

    '''

    # import sizes
    n = len(smiles)
    n_atom_features = features.num_atom_features()
    n_bond_features = features.num_bond_features()

    # preallocate atom tensor with 0's and bond tensor with -1 (because of 0 index)
    atom_tensor = np.zeros((n, max_atoms, n_atom_features))
    bond_tensor = np.zeros((n, max_atoms, max_degrees, n_bond_features))
    edge_tensor = -np.ones((n, max_atoms, max_degrees), dtype=int)

    for mol_ix, s in enumerate(smiles):

        #load mol, atoms and bonds
        mol = Chem.MolFromSmiles(s)
        assert mol is not None, 'Could not parse smiles {}'.format(s)        
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        assert len(atoms) <= max_atoms, 'too many atoms in molecule'

        rdkit_ix_lookup = {}
        connectivity_mat = {}

        for atom_ix, atom in enumerate(atoms):
            # write atom features
            atom_tensor[mol_ix, atom_ix, : n_atom_features] = features.atom_features(atom)

            # store entry in idx
            rdkit_ix_lookup[atom.GetIdx()] = atom_ix

        # preallocate array with neighbour lists (indexed by atom)
        connectivity_mat = [ [] for _ in atoms]

        for bond in bonds:
            # lookup atom ids
            a1_ix = rdkit_ix_lookup[bond.GetBeginAtom().GetIdx()]
            a2_ix = rdkit_ix_lookup[bond.GetEndAtom().GetIdx()]            

            # lookup how many neighbours are encoded yet
            a1_neigh = len(connectivity_mat[a1_ix])
            a2_neigh = len(connectivity_mat[a2_ix])

            # store bond features
            bond_features = np.array(features.bond_features(bond), dtype=int)
            bond_tensor[mol_ix, a1_ix, a1_neigh, :] = bond_features
            bond_tensor[mol_ix, a2_ix, a2_neigh, :] = bond_features

            #add to connectivity matrix
            connectivity_mat[a1_ix].append(a2_ix)
            connectivity_mat[a2_ix].append(a1_ix)

        #store connectivity matrix
        for a1_ix, neighbours in enumerate(connectivity_mat):
            degree = len(neighbours)
            assert degree <= max_degrees, 'too many neighbours for atom'
            edge_tensor[mol_ix, a1_ix, : degree] = neighbours

    return atom_tensor, bond_tensor, edge_tensor