import itertools
from accent_restoration.utilities.utils import remove_accent
from accent_restoration.utilities.encoder_decoder import alphabet, CharacterCodec
import numpy as np
import string
from accent_restoration.utilities.config import MAXLEN, INVERT


codec = CharacterCodec(alphabet, MAXLEN)

def gen_stream(ngrams):
    """ generate an infinite stream of (input, output) pair from phrases """
    while True:
        for s in ngrams:
            output_s = s + ' ' * (MAXLEN - len(s))
            input_s = remove_accent(output_s)    
            input_s = input_s[::-1] if INVERT else input_s
            input_vec = codec.try_encode(input_s)
            output_vec = codec.try_encode(output_s)
            if input_vec is not None and output_vec is not None:
                yield input_vec, output_vec

def gen_batch(it, size):
    """ batch the input iterator to iterator of list of given size"""
    for _, group in itertools.groupby(enumerate(it), lambda x: x[0] // size):
        yield list(zip(*group))[1]


def gen_data(ngrams, batch_size=128):
    """ generate infinite X, Y array of batch_size from given phrases """
    for batch in gen_batch(gen_stream(ngrams), size=batch_size):
        # we need to return X, Y array from one batch, which is a list of (x, y) pair
        X, Y = zip(*batch)
        yield np.array(X), np.array(Y)
