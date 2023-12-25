import pickle
import numpy as np
from scipy import sparse
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from ..dataset import Dataset
from ..datasets.tokenizer import VocabularyTokenizer


class MoleculeGraphDataset(Dataset):
    def __init__(self, logger, name, dfs, 
        df, col, path_voc, error='drop', path_length=None, is_canonical=False,
            get_canonical=False, **kwargs):
        """
        SMILESを読み込み, 隣接行列などを出力するDataset
        
        Parameters
        ----------
        df (str): SMILESデータのあるDataFrame
        col (str): SMILESの行
        path_voc (str): SMILESのトークン用のvocabulary
        error (str): SMILESデータ作成中にエラーが起きた時の対応。
            'drop': 削除。idxを変更する。
            'drop_keep_idx': 削除。idxは変更しない。
            'raise': エラーを出す。   
        path_lengths (str, optional): データの「長さ(bucketing用)」を得る。
            無い場合はSMILES長さをbucketing長さとする。
        path_voc (str): SMILESのトークン化用vocabulary.
        is_canonical (bool): Trueの場合, SMILESをそのままcanonical SMILESとして使用する。
        get_canonical (bool): canonical SMILES tokenをbatchに入れるかどうか
        """
        self.smiles = dfs[df][col].values
        with open(path_voc) as f:
            self.toker = VocabularyTokenizer(f.read().splitlines())
        self.error = error
        if path_length is None:
            self.lengths = torch.tensor([len(smi) for smi in self.smiles])
        else:
            with open(path_length, 'rb') as f:
                self.lengths = torch.tensor(pickle.load(f))
        self.is_can = is_canonical
        self.get_can = get_canonical

    def make_batch(self, batch, idx, device):
        mols = []
        valid_idxs = []
        asizes = []
        for i in idx:
            try:
                smi = self.smiles[i]
                mol = Chem.MolFromSmiles(smi)
                mol = AllChem.AddHs(mol)
                r = AllChem.EmbedMolecule(mol, randomSeed=0, enforceChirality=False)
                if r != 0: raise ValueError('Bad conformer ID')
                rdmolops.AssignAtomChiralTagsFromStructure(mol)
                mols.append(mol)
                valid_idxs.append(i)
                asizes.append(mol.GetNumAtoms())
                max_ssize = max(max_ssize, len(can_token))
            except Exception as e:
                if self.error in ['drop', 'drop_keep_idx']:
                    continue
                elif self.error == 'raise':
                    raise ValueError(e)
        max_asize = max(asizes)
        batch_size = len(mols)

        atypes = np.zeros((batch_size, max_asize), dtype=int)
        chirals = np.zeros((batch_size, max_asize), dtype=int)
        bonds = np.zeros((batch_size, max_asize, max_asize), dtype=float)
        coords = np.zeros((batch_size, max_asize, 3), dtype=float)
        for i_point, (mol, asize) in enumerate(zip(mols, asizes)):
            for i_atom, atom in enumerate(mol.GetAtoms()):
                atypes[i_point, i_atom] = atom.GetAtomicNum()
            for i_atom, direction in Chem.FindMolChiralCenters(mol, includeUnassigned=True):
                if direction == 'R':
                    chirals[i_point][i_atom] = 1
                elif direction == 'S':
                    chirals[i_point][i_atom] = 2
                elif direction == '?':
                    chirals[i_point][i_atom] = 3
            bonds[i_point, :asize, :asize] = rdmolops.GetAdjacencyMatrix(mol, useBO=1)

            coords[i_point, :asize] = mol.GetConformer().GetPositions()
        bonds[bonds == 1.5] = 5
        bonds = bonds.astype(int)
        batch['idx'] = np.array(valid_idxs, dtype=int)
        batch['atom_type'] = torch.tensor(atypes, device=device)
        batch['chiral'] = torch.tensor(chirals, device=device)
        batch['bond'] = torch.tensor(bonds, device=device)
        batch['coordinate'] = torch.tensor(coords, device=device).to(torch.float)

        if self.get_can:
            can_tokens = []
            max_ctsize = 0
            if self.is_can:
                for i in valid_idxs:
                    token = self.toker.tokenize(self.smiles[i])
                    can_tokens.append(token)
                    max_ctsize = max(max_ctsize, len(token))
            else:
                for mol in mols:
                    mol = AllChem.RemoveHs(mol)
                    token = self.toker.tokenize(Chem.MolToSmiles(mol))
                    can_tokens.append(token)
                    max_ctsize = max(max_ctsize, len(token))
            canonical = np.full((batch_size, max_ctsize), fill_value=self.toker.pad_token,
                dtype=int)
            for i_point, can_token in enumerate(can_tokens):
                canonical[i_point, :len(can_token)] = can_token
            batch['canonical'] = torch.tensor(canonical, device=device)