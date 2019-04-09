#!/usr/bin/env python
# coding: utf-8

import string
import numpy as np
from accent_restoration.utilities.utils import plain_char_map
alphabet = set(' ' + string.ascii_lowercase + string.digits + ''.join(plain_char_map.keys()))

class CharacterCodec(object):
    def __init__(self, alphabet, maxlen):
        self.alphabet = list(sorted(alphabet))
        self.index_alphabet = dict((c, i) for i, c in enumerate(self.alphabet))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.alphabet)))
        for i, c in enumerate(C[:maxlen]):
            X[i, self.index_alphabet[c]] = 1
        return X
    
    def try_encode(self, C, maxlen=None):
        try:
            return self.encode(C, maxlen)
        except KeyError:
            return self.encode(' ', maxlen)

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.alphabet[x] for x in X)

