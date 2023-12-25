"""
230822作成
tokenizer.pyのVocabularyTokenizer2から改変。
voc_lensが<padding>などの長さを含まないようにしたほかは同じ。
"""

import numpy as np

class VocabularyTokenizer():
    def __init__(self, vocs):
        """
        Parameters
        ----------
        vocs: array-like of str
        """
        vocs = sorted(list(vocs), key=lambda x:len(x), reverse=True)
        vocs_with_specials = ['<padding>', '<start>', '<end>'] + vocs
        self.voc_lens = np.sort(np.unique([len(voc) for voc in vocs]))[::-1]
        self.min_voc_len = self.voc_lens[-1]
        self.pad_token = 0
        self.start_token = 1
        self.end_token = 2
        self.voc2tok = {voc: tok for tok, voc in enumerate(vocs_with_specials)}
        self.tok2voc = np.array(vocs_with_specials)

    def tokenize(self, string):
        string_left = string
        toks = [self.start_token]
        while len(string_left) > 0:
            for voc_len in self.voc_lens:
                if string_left[:voc_len] in self.voc2tok:
                    toks.append(self.voc2tok[string_left[:voc_len]])
                    string_left = string_left[voc_len:]
                    break
                if voc_len == self.min_voc_len:
                    raise KeyError(f"Unknown keyward '{string_left}' in {string}")
        toks.append(self.end_token)
        return toks

    def detokenize(self, toks):
        """
        Parameters
        ----------
        toks: array_like of int

        Returns
        -------
        string: str
            detokenized string.
        """
        string = ""
        for tok in toks:
            if tok == self.end_token:
                break
            elif tok != self.start_token:
                string += self.tok2voc[tok]
        return string

    @property
    def voc_size(self):
        return len(self.tok2voc)

class OrderedTokenizer():
    """
    __init__のvocsをそのままの順でtoken対応させる。
    """
    def __init__(self, vocs):
        vocs[0] # check order exists in vocs
        vocs = ['<padding>', '<start>', '<end>'] + list(vocs)
        self.voc_lens = np.sort(np.unique([len(voc) for voc in vocs]))[::-1]
        self.min_voc_len = self.voc_lens[-1]
        self.pad_token = 0
        self.start_token = 1
        self.end_token = 2
        self.voc2tok = {voc: tok for tok, voc in enumerate(vocs)}
        self.tok2voc = np.array(vocs)

    def tokenize(self, string):
        string_left = string
        toks = [self.start_token]
        while len(string_left) > 0:
            for voc_len in self.voc_lens:
                if string_left[:voc_len] in self.voc2tok:
                    toks.append(self.voc2tok[string_left[:voc_len]])
                    string_left = string_left[voc_len:]
                    break
                if voc_len == self.min_voc_len:
                    raise KeyError(string)
        toks.append(self.end_token)
        return toks

    def detokenize(self, toks):
        """
        Parameters
        ----------
        toks: array_like of int

        Returns
        -------
        string: str
            detokenized string.
        """
        string = ""
        for tok in toks:
            if tok == self.end_token:
                break
            elif tok != self.start_token:
                string += self.tok2voc[tok]
        return string

    @property
    def voc_size(self):
        return len(self.tok2voc)