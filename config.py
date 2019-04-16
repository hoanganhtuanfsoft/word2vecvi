class preprocessing(object):
    teen_code_file = "teen_code.csv"

    minimum_number_of_words = 10

class training_word2vec(object):
    FASTTEXT_MODEL_PATH = "./model/word2vec_fasttext.model"

    SKIPGRAM_MODEL_PATH = "./model/word2vec_skipgram.model"

    WINDOW_SIZE = 10
    VECTOR_DIMENSIONS = 200
    NUMBER_OF_THREADS = 8
    EPOCHS = 10
    MIN_COUNT = 5

    TRAINING_DATA = "./data/data_offical.txt"

    FASTTEXT = 2
    SKIPGRAM = 1
    ALL = 0

    TRAINING_MODE = FASTTEXT
    TESTING_MODE = FASTTEXT



