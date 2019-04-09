import os
import pandas as pd
import string
from pyvi import ViTokenizer
from gensim.models.fasttext import FastText
from gensim.models import Word2Vec
from config import training_word2vec


def implement_fasttext(train_data = None, saved_path = training_word2vec.FASTTEXT_MODEL_PATH):
    if train_data is not None:
        print("[INFO][TRAINING W2V][FASTTEXT] BEGIN")
        model_fasttext = FastText(size=training_word2vec.VECTOR_DIMENSIONS, \
                                    window=training_word2vec.WINDOW_SIZE, \
                                        min_count=training_word2vec.MIN_COUNT, \
                                        workers=training_word2vec.NUMBER_OF_THREADS, \
                                            sg=1, \
                                            iter=training_word2vec.EPOCHS)
        model_fasttext.build_vocab(train_data)

        model_fasttext.train(train_data, total_examples=model_fasttext.corpus_count, epochs=model_fasttext.iter)
        
        model_fasttext.wv.save(saved_path)

        print("[INFO][TRAINING W2V][FASTTEXT] END")
        return True
    
    return False

def implement_w2v_skipgram(train_data = None, saved_path = training_word2vec.SKIPGRAM_MODEL_PATH):
    if train_data is not None:
        print("[INFO][TRAINING W2V][SKIPGRAM] BEGIN")
        model = Word2Vec(train_data, size=training_word2vec.VECTOR_DIMENSIONS, \
                                        window=training_word2vec.WINDOW_SIZE, \
                                            min_count=training_word2vec.MIN_COUNT, \
                                                workers=training_word2vec.NUMBER_OF_THREADS, sg=1, \
                                                    iter = training_word2vec.EPOCHS)
        model.wv.save(saved_path)
        print("[INFO][TRAINING W2V][SKIPGRAM] END")
        return True
    
    return False

def read_data(path):
    traindata = []
    sents = open(path, 'r').readlines()
    for sent in sents:
        traindata.append(sent.split())
    return traindata

if __name__ == '__main__':
    # Load data 
    train_data = read_data(training_word2vec.TRAINING_DATA)

    # Train word embeddings
    if training_word2vec.TRAINING_MODE == training_word2vec.FASTTEXT:
        implement_fasttext(train_data=train_data)

    if training_word2vec.TRAINING_MODE == training_word2vec.SKIPGRAM:
        implement_w2v_skipgram(train_data=train_data)
    
    if training_word2vec.TRAINING_MODE == training_word2vec.ALL:
        implement_fasttext(train_data=train_data)
        implement_w2v_skipgram(train_data=train_data)
        




 