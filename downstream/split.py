"""
GEM, Uni-Molで使われていたscaffold splitting, random scaffold splittingを実装。
"""
from collections import defaultdict

import numpy as np
import rdkit
from rdkit.Chem.Scaffolds import MurckoScaffold
print(f"rdkit: {rdkit.__version__}")

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles

    Args:
        smiles: smiles sequence
        include_chirality: Default=False
    Return: 
        the scaffold of the given smiles.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

def scaffold_split(dataset, 
        frac_train=None, 
        frac_valid=None, 
        frac_test=None):
    """
    Args:
        dataset(list[str]): list of smiles
        frac_train(float): the fraction of data to be used for the train split.
        frac_valid(float): the fraction of data to be used for the valid split.
        frac_test(float): the fraction of data to be used for the test split.
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    N = len(dataset)

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i in range(N):
        scaffold = generate_scaffold(dataset[i], include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid test indices
    train_cutoff = frac_train * N
    valid_cutoff = (frac_train + frac_valid) * N
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    # get train, valid test indices
    train_cutoff = frac_train * N
    valid_cutoff = (frac_train + frac_valid) * N
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    return train_idx, valid_idx, test_idx

def random_scaffold_split(dataset, 
        frac_train=None, 
        frac_valid=None, 
        frac_test=None,
        seed=None):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    N = len(dataset)

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind in range(N):
        scaffold = generate_scaffold(dataset[ind], include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    return train_idx, valid_idx, test_idx
    
