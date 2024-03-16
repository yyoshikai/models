import pickle

import torch
import numpy as np

from ..dataset import Dataset

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
# get_atom_fdimより
ATOM_FDIM, BOND_FDIM = ATOM_FDIM+18, ATOM_FDIM+BOND_FDIM+18


class GroverDataset(Dataset):
    def __init__(self, logger, name, dfs, data_path, bond_drop_rate):
        super().__init__(logger, name, dfs)
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

    def make_batch(self, batch, idx, device):

        # Start n_atoms and n_bonds at 1 b/c zero padding
        batch_n_atom = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        batch_n_bond = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        batch_a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        batch_b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        batch_f_atoms = [[0] * ATOM_FDIM]  # atom features
        batch_f_bonds = [[0] * BOND_FDIM]  # combined atom/bond features
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
        return len(self.f_atomss)

def onek_encoding_unk(value: int, choices):
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    if min(choices) < 0:
        index = value
    else:
        index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

# 重かったので, その場で生成するようにする。
class Grover2Dataset(Dataset):
    def __init__(self, logger, name, dfs, df, col, bond_drop_rate):
        super().__init__(logger, name, dfs)

        from rdkit import Chem
        self.chem = Chem
        
        self.bond_drop_rate = bond_drop_rate
        self.smiles = dfs[df][col].values

        self.hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        self.hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),"
            "n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        self.acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        self.basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);"
            "!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")

    def make_batch(self, batch, idx, device):
        
        a2bs = []
        b2as = []
        b2revbs = []
        f_atomss = []
        f_bondss = []
        n_atomss = []
        n_bondss = []

        for i in idx:
            smiles = self.smiles[i]
            n_atoms = 0  # number of atoms
            n_bonds = 0  # number of bonds
            f_atoms = []  # mapping from atom index to atom features
            f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
            a2b = []  # mapping from atom index to incoming bond indices
            b2a = []  # mapping from bond index to the index of the atom the bond is coming from
            b2revb = []  # mapping from bond index to the index of the reverse bond

            # Convert smiles to molecule
            mol = self.chem.MolFromSmiles(smiles)

            hydrogen_donor_match = sum(mol.GetSubstructMatches(self.hydrogen_donor), ())
            hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self.hydrogen_acceptor), ())
            acidic_match = sum(mol.GetSubstructMatches(self.acidic), ())
            basic_match = sum(mol.GetSubstructMatches(self.basic), ())
            ring_info = mol.GetRingInfo()


            # fake the number of "atoms" if we are collapsing substructures
            n_atoms = mol.GetNumAtoms()

            # Get atom features
            for _, atom in enumerate(mol.GetAtoms()):
                f_atoms.append(self.atom_features(atom, hydrogen_acceptor_match, hydrogen_donor_match, acidic_match, basic_match, ring_info))
            f_atoms = [f_atoms[i] for i in range(n_atoms)]

            for _ in range(n_atoms):
                a2b.append([])

            # Get bond features
            for a1 in range(n_atoms):
                for a2 in range(a1 + 1, n_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue

                    if self.bond_drop_rate > 0:
                        if np.random.binomial(1, self.bond_drop_rate):
                            continue

                    f_bond = self.bond_features(bond)

                    # Always treat the bond as directed.
                    f_bonds.append(f_atoms[a1] + f_bond)
                    f_bonds.append(f_atoms[a2] + f_bond)

                    # Update index mappings
                    b1 = n_bonds
                    b2 = b1 + 1
                    a2b[a2].append(b1)  # b1 = a1 --> a2
                    b2a.append(a1)
                    a2b[a1].append(b2)  # b2 = a2 --> a1
                    b2a.append(a2)
                    b2revb.append(b2)
                    b2revb.append(b1)
                    n_bonds += 2
            
            a2bs.append(a2b)
            b2as.append(b2a)
            b2revbs.append(b2revb)
            f_atomss.append(f_atoms)
            f_bondss.append(f_bonds)
            n_atomss.append(n_atoms)
            n_bondss.append(n_bonds)

        # Start n_atoms and n_bonds at 1 b/c zero padding
        batch_n_atom = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        batch_n_bond = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        batch_a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        batch_b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        batch_f_atoms = [[0] * ATOM_FDIM]  # atom features
        batch_f_bonds = [[0] * BOND_FDIM]  # combined atom/bond features
        batch_a2b = [[]]  # mapping from atom index to incoming bond indices
        batch_b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        batch_b2revb = [0]  # mapping from bond index to the index of the reverse bond

        for i in range(len(idx)):
            batch_f_atoms.extend(f_atomss[i])
            batch_f_bonds.extend(f_bondss[i])

            for a in range(n_atomss[i]):
                batch_a2b.append([b + batch_n_bond for b in a2bs[i][a]])

            for b in range(n_bondss[i]):
                batch_b2a.append(n_atomss[i] + b2as[i][b])
                batch_b2revb.append(batch_n_bond + b2revbs[i][b])

            batch_a_scope.append((batch_n_atom, n_atomss[i]))
            batch_b_scope.append((batch_n_bond, n_bondss[i]))
            batch_n_atom += n_atomss[i]
            batch_n_bond += n_bondss[i]

        # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        max_num_bonds = max(1, max(len(in_bonds) for in_bonds in batch_a2b))
        batch_f_atoms = torch.FloatTensor(batch_f_atoms).to(device)
        batch_f_bonds = torch.FloatTensor(batch_f_bonds).to(device)
        batch_a2b = torch.LongTensor([batch_a2b[a] + [0] * (max_num_bonds - len(batch_a2b[a])) 
            for a in range(batch_n_atom)]).to(device)
        batch_b2a = torch.LongTensor(batch_b2a).to(device)
        batch_b2revb = torch.LongTensor(batch_b2revb).to(device)
        batch_a2a = batch_b2a[batch_a2b]  # only needed if using atom messages

        batch['f_atoms'] = batch_f_atoms
        batch['f_bonds'] = batch_f_bonds
        batch['a2b'] = batch_a2b
        batch['b2a'] = batch_b2a
        batch['b2revb'] = batch_b2revb
        batch['a_scope'] = batch_a_scope
        batch['b_scope'] = batch_b_scope
        batch['a2a'] = batch_a2a

        return batch
    
        
    def atom_features(self, atom, hydrogen_acceptor_match, hydrogen_donor_match, acidic_match, basic_match, ring_info):
        features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
                    onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
                    onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
                    onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
                    onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
                    onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
                    [1 if atom.GetIsAromatic() else 0] + \
                    [atom.GetMass() * 0.01]
        atom_idx = atom.GetIdx()
        features = features + \
                    onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                    [atom_idx in hydrogen_acceptor_match] + \
                    [atom_idx in hydrogen_donor_match] + \
                    [atom_idx in acidic_match] + \
                    [atom_idx in basic_match] + \
                    [ring_info.IsAtomInRingOfSize(atom_idx, 3),
                    ring_info.IsAtomInRingOfSize(atom_idx, 4),
                    ring_info.IsAtomInRingOfSize(atom_idx, 5),
                    ring_info.IsAtomInRingOfSize(atom_idx, 6),
                    ring_info.IsAtomInRingOfSize(atom_idx, 7),
                    ring_info.IsAtomInRingOfSize(atom_idx, 8)]
        return features

    def bond_features(self, bond):
        if bond is None:
            fbond = [1] + [0] * (BOND_FDIM - 1)
        else:
            bt = bond.GetBondType()
            fbond = [
                0,  # bond is not None
                bt == self.chem.rdchem.BondType.SINGLE,
                bt == self.chem.rdchem.BondType.DOUBLE,
                bt == self.chem.rdchem.BondType.TRIPLE,
                bt == self.chem.rdchem.BondType.AROMATIC,
                (bond.GetIsConjugated() if bt is not None else 0),
                (bond.IsInRing() if bt is not None else 0)
            ]
            fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
        return fbond


    def __len__(self):
        return len(self.smiles)
