class preprocessing(object):
    teen_code_file = "teen_code.txt"

    minimum_number_of_words = 6

class training_word2vec(object):
    FASTTEXT_MODEL_PATH = "./model/word2vec_fasttext.model"

    SKIPGRAM_MODEL_PATH = "./model/word2vec_skipgram.model"

    WINDOW_SIZE = 5
    VECTOR_DIMENSIONS = 400
    NUMBER_OF_THREADS = 4
    EPOCHS = 10
    MIN_COUNT = 2

    TRAINING_DATA = "./data/data_ver02.txt"

    FASTTEXT = 2
    SKIPGRAM = 1
    ALL = 0

    TRAINING_MODE = SKIPGRAM
    TESTING_MODE = SKIPGRAM



