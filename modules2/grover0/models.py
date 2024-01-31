"""
The GROVER models for pretraining, finetuning and fingerprint generating.
"""
from typing import List, Dict
from torch import nn as nn
from .layers import Readout

class AtomVocabPrediction(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, embeddings):
        return self.logsoftmax(self.linear(embeddings))


class BondVocabPrediction(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.linear_rev = nn.Linear(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, embeddings):
        nm_bonds = embeddings.shape[0]  # must be an odd number
        ids1 = [0] + list(range(1, nm_bonds, 2))
        ids2 = list(range(0, nm_bonds, 2))
        logits = self.linear(embeddings[ids1]) + self.linear_rev(embeddings[ids2])
        
        return self.logsoftmax(logits)

class FunctionalGroupPrediction(nn.Module):
    def __init__(self, fg_size, hidden_size):
        super(FunctionalGroupPrediction, self).__init__()
        first_linear_dim = hidden_size

        # In order to retain maximal information in the encoder, we use a simple readout function here.
        self.readout = Readout(rtype="mean", hidden_size=hidden_size)
        # We have four branches here. But the input with less than four branch is OK.
        # Since we use BCEWithLogitsLoss as the loss function, we only need to output logits here.
        self.linear_atom_from_atom = nn.Linear(first_linear_dim, fg_size)
        self.linear_atom_from_bond = nn.Linear(first_linear_dim, fg_size)
        self.linear_bond_from_atom = nn.Linear(first_linear_dim, fg_size)
        self.linear_bond_from_bond = nn.Linear(first_linear_dim, fg_size)

    def forward(self, embeddings: Dict, ascope: List, bscope: List):
        preds_bond_from_atom = self.linear_bond_from_atom(self.readout(embeddings["bond_from_atom"], bscope))
        preds_bond_from_bond = self.linear_bond_from_bond(self.readout(embeddings["bond_from_bond"], bscope))
        preds_atom_from_atom = self.linear_atom_from_atom(self.readout(embeddings["atom_from_atom"], ascope))
        preds_atom_from_bond = self.linear_atom_from_bond(self.readout(embeddings["atom_from_bond"], ascope))

        return {"atom_from_atom": preds_atom_from_atom, "atom_from_bond": preds_atom_from_bond,
                "bond_from_atom": preds_bond_from_atom, "bond_from_bond": preds_bond_from_bond}

