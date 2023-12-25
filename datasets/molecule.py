import sys
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
print(f"[WARNING] RDKit version: {rdkit.__version__}", file=sys.stderr)

def randomize(mol, can, rstate, n_trial=100):
    nums = np.arange(mol.GetNumAtoms())
    for i_trial in range(n_trial):
        rstate.shuffle(nums)
        mol = Chem.RenumberAtoms(mol, nums.tolist())
        ran = Chem.MolToSmiles(mol, canonical=False)
        if can == Chem.MolToSmiles(Chem.MolFromSmiles(ran)):
            break
    else:
        raise ValueError
    return ran

REMOVER = SaltRemover()
def salt_remove(mol):
    mol = REMOVER.StripMol(mol, dontRemoveEverything=True)
    t_smile = Chem.MolToSmiles(mol, isomericSmiles=True)
    if '.' in t_smile:
        mol_frags = Chem.GetMolFrags(mol, asMols=True)
        largest_size = 0
        for frag in mol_frags:
            size = frag.GetNumAtoms()
            if size > largest_size:
                mol = frag
                largest_size = size
    return mol

ORGANIC_ATOMS = {1,5,6,7,8,9,15,16,17,35,53}
def has_irregular_atom(mol, regular_atoms=ORGANIC_ATOMS):
    atoms = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
    return len(atoms - regular_atoms) > 0


