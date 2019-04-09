#!/usr/bin/env python
# coding: utf-8

from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, LSTM
from keras.callbacks import Callback
from accent_restoration.utilities.config import MAXLEN
from accent_restoration.utilities.encoder_decoder import alphabet

HIDDEN_SIZE = 256

model = Sequential()
model.add(LSTM(HIDDEN_SIZE, input_shape=(MAXLEN, len(alphabet))))
model.add(RepeatVector(MAXLEN))
model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
model.add(TimeDistributed(Dense(len(alphabet))))
model.add(Activation('softmax'))
# configuration for compiling
LOSS = 'categorical_crossentropy'
OPTIMIZER = 'adam'
METRICS = ['accuracy']
WEIGHTS_NAME = 'epoch_9_full.hdf5'
