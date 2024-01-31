import pickle

import torch
import pandas as pd

from models.dataset import Dataset, dataset_type2class

MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [0, 1, 2, 3, 4], # Chem.rdchem.HybridizationType
}

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14

class GroverDataset(Dataset):
    def __init__(self, logger, name, dfs, data_path, atom_vocab_path, bond_vocab_path, bond_drop_rate):
        super().__init__(logger, name, dfs)
        self.smiles = pd.read_csv(f"exampledata/pretrain/tryout.csv")['smiles'].values

        self.bond_drop_rate = bond_drop_rate

        with open(f"{data_path}/a2b.pkl", 'rb') as f:
            self.a2bs = pickle.load(f)
        with open(f"{data_path}/b2a.pkl", 'rb') as f:
            self.b2as = pickle.load(f)
        with open(f"{data_path}/b2revb.pkl", 'rb') as f:
            self.b2revbs = pickle.load(f)
        with open(f"{data_path}/f_atoms.pkl", 'rb') as f:
            self.f_atomss = pickle.load(f)
        with open(f"{data_path}/f_bonds.pkl", 'rb') as f:
            self.f_bondss = pickle.load(f)
        with open(f"{data_path}/feature.pkl", 'rb') as f:
            self.featuress = pickle.load(f)
        with open(f"{data_path}/n_atoms.txt", 'r') as f:
            self.n_atoms = [int(n) for n in f.read().splitlines()]
        with open(f"{data_path}/n_bonds.txt", 'r') as f:
            self.n_bonds = [int(n) for n in f.read().splitlines()]

        self.atom_fdim = ATOM_FDIM
        self.bond_fdim = BOND_FDIM + self.atom_fdim
        
        

    def make_batch(self, batch, idx, device):

        smiles_batch = self.smiles[idx]

        # Start n_atoms and n_bonds at 1 b/c zero padding
        batch_n_atom = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        batch_n_bond = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        batch_a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        batch_b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        batch_f_atoms = [[0] * self.atom_fdim]  # atom features
        batch_f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        batch_a2b = [[]]  # mapping from atom index to incoming bond indices
        batch_b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        batch_b2revb = [0]  # mapping from bond index to the index of the reverse bond

        for i in idx:
            batch_f_atoms.extend(self.f_atomss[i])
            batch_f_bonds.extend(self.f_bondss[i])

            for a in range(self.n_atoms[i]):
                batch_a2b.append([b + batch_n_bond for b in self.a2bs[i][a]])

            for b in range(self.n_bonds[i]):
                batch_b2a.append(self.n_atoms[i] + self.b2as[i][b])
                batch_b2revb.append(batch_n_bond + self.b2revbs[i][b])

            batch_a_scope.append((batch_n_atom, self.n_atoms[i]))
            batch_b_scope.append((batch_n_bond, self.n_bonds[i]))
            batch_n_atom += self.n_atoms[i]
            batch_n_bond += self.n_bonds[i]

        # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        max_num_bonds = max(1, max(len(in_bonds) for in_bonds in batch_a2b))

        batch_f_atoms = torch.FloatTensor(batch_f_atoms).to(device)
        batch_f_bonds = torch.FloatTensor(batch_f_bonds).to(device)
        batch_a2b = torch.LongTensor([batch_a2b[a] + [0] * (max_num_bonds - len(batch_a2b[a])) 
            for a in range(batch_n_atom)]).to(device)
        batch_b2a = torch.LongTensor(batch_b2a).to(device)
        batch_b2revb = torch.LongTensor(batch_b2revb).to(device)
        batch_a2a = batch_b2a[batch_a2b]  # only needed if using atom messages

        batch_b_scope = torch.LongTensor(batch_b_scope).to(device)

        batch['f_atoms'] = batch_f_atoms
        batch['f_bonds'] = batch_f_bonds
        batch['a2b'] = batch_a2b
        batch['b2a'] = batch_b2a
        batch['b2revb'] = batch_b2revb
        batch['a_scope'] = batch_a_scope
        batch['b_scope'] = batch_b_scope
        batch['a2a'] = batch_a2a

        return batch

    def __len__(self):
        return len(self.smiles)
dataset_type2class['grover'] = GroverDataset