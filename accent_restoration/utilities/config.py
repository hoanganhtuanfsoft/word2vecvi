# Set up hyperparameters
# maximum string length to train and predict
# this is set based on our ngram length break down above
MAXLEN = 32

# minimum string length to consider
MINLEN = 3

# inverting the input generally help with accuracy
INVERT = True

# mini batch size
BATCH_SIZE = 4096

# ngrams
NGRAM = 5
