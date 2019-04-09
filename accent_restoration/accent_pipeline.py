#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from nltk.util import ngrams
import re
import regex
import os.path
from collections import Counter

# load utilities from .py file in current directory
from accent_restoration.utilities.utils import remove_accent, plain_char_map, compound_unicode
from accent_restoration.utilities.pre_process import extract_phrases
from accent_restoration.utilities.config import MAXLEN, INVERT, NGRAM
from accent_restoration.utilities.generator import codec
# load lebinh model
from accent_restoration.model.lebinh import model, LOSS, OPTIMIZER, METRICS, WEIGHTS_NAME
# compile model
model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
# load model weights
# get the abspath of current pipeline file
script_path = os.path.dirname(os.path.abspath( __file__ ))
# get the abspath of model's weights
model_weights_path = os.path.join(script_path, 'model', WEIGHTS_NAME)
# load the weights
model.load_weights(model_weights_path)

# MAIN PARTS OF THE PIPELINE IS BELOW, HOLD ON FOR A WHILE FORKS
def guess(ngram, INVERT=True):
    text = ' '.join(ngram)
    text += ' ' * (MAXLEN - len(text))
    if INVERT:
        text = text[::-1]
    preds = model.predict_classes(np.array([codec.try_encode(text)]), verbose=0)
    return codec.decode(preds[0], calc_argmax=False).rstrip(' ')

def add_accent(text):
    ngrams_list = list(ngrams(text.lower().split(), n=NGRAM, pad_right=True, right_pad_symbol=' '))
    guessed_ngrams = list(guess(ngram) for ngram in ngrams_list)
    candidates = [Counter() for _ in range(len(guessed_ngrams) + NGRAM - 1)]
    for nid, ngram in enumerate(guessed_ngrams):
        for wid, word in enumerate(re.split(' +', ngram)):
            candidates[nid + wid].update([word])
    output = ' '.join(c.most_common(1)[0][0] for c in candidates if len(c.most_common(1))>0)
    return output

def accent_restore(text):
    # lower and strip inpit text
    text = text.lower().strip()
    # convert from compound unicode to precomposed unicode
    text = compound_unicode(text)
    
    # replace underscore with space because in can confuse the \w regex
    text = re.sub('_',' ',text) # replace '_' by ' '
    
    # remove non latin alphabel character (chinesse, japanese, etc - wil support later!!! - including \s)
    # very troublesome function since it also removes some special characters like =, (, )
    # must list special characters we want to retain manually in the second parentheses
    text = regex.sub(u'[^\p{Latin}--{ \n,.\t()}!]', '', text)
    
    # get the list of non alphanumeric characters and their corresponding places in original phrase to concatenate later
    non_alphanumeric_place_start = []
    non_alphanumeric_value = []
    non_alphanumeric_regex = re.compile(' {0,1}[^\w ]+ {0,1}')
    for m in non_alphanumeric_regex.finditer(text):
        non_alphanumeric_place_start.append(m.start()) # get the starting places of non alphanumeric characters
        non_alphanumeric_value.append(m.group()) # get the values of non alphanumeric characters
        
    # get the list of alphanumeric characters and their places
    alphanumeric_place_start = []
    alphanumeric_place_value = []
    alphanumeric_regex = re.compile('\w[\w ]+')
    for n in alphanumeric_regex.finditer(text):
        alphanumeric_place_start.append(n.start()) # get the starting places of non alphanumeric characters
        alphanumeric_place_value.append(n.group()) # get the starting places of non alphanumeric characters
    
    # strip all accents before feeding to model
    text = remove_accent(text)

    # ADD ACCENTS BY MODEL
    phrases = extract_phrases(text)
    phrases = [add_accent(p) for p in phrases]
    
    # CONCATENATING STEPS
    # TODO: new concatenating mechanism
    # empty dictionary to put the place and value
    place_value_dict = {}
    return_phrase = ''
    # zip non-alpha numeric place and value to list
    non_alphanumeric = dict(zip(non_alphanumeric_place_start, non_alphanumeric_value))
    # zip alpha numeric place and value to list
    alphanumeric = dict(zip(alphanumeric_place_start, phrases))
    # update the overall dictionary
    place_value_dict.update(non_alphanumeric)
    place_value_dict.update(alphanumeric)
    # looping through the sorted overall dictionary and concatenate
    for place in sorted(list(place_value_dict.keys())):
        return_phrase = return_phrase + place_value_dict[place]
    
    # strip extra white spaces
    return_phrase = re.sub(' [ ]+',' ',return_phrase)
    return_phrase = re.sub(' $','',return_phrase)
    # TEMPORARILY: model not get use to joy or gonjoy terms so it might translate wrong and need to replace by regex
    # WILL REMOVE LATER!
    return_phrase = re.sub('jo.','joy',return_phrase)
    return_phrase = re.sub('jò.','joy',return_phrase)
    return_phrase = re.sub('jó.','joy',return_phrase)
    return_phrase = re.sub('jỏ.','joy',return_phrase)
    return_phrase = re.sub('jõ.','joy',return_phrase)
    return_phrase = re.sub('jọ.','joy',return_phrase)
    return_phrase = re.sub('gonj..','gonjoy',return_phrase)
    return return_phrase




