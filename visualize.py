import gensim.models.keyedvectors as word2vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from config import training_word2vec

import os
if training_word2vec.TESTING_MODE == training_word2vec.SKIPGRAM:
    model = word2vec.KeyedVectors.load(training_word2vec.SKIPGRAM_MODEL_PATH)
else:
    model = word2vec.KeyedVectors.load(training_word2vec.FASTTEXT_MODEL_PATH)

pathfile = './words'
with open(pathfile, 'r') as f:
    words = f.readlines()
    words = [word.strip() for word in words]

words_np = []
words_label = []

for word in model.vocab.keys():
    if word in words:
        words_np.append(model[word])
        words_label.append(word)

pca = PCA(n_components=2)
pca.fit(words_np)
reduced = pca.transform(words_np)


def visualize():
    fig, ax = plt.subplots()

    for index, vec in enumerate(reduced):
        x, y = vec[0], vec[1]

        ax.scatter(x, y)
        ax.annotate(words_label[index], xy=(x, y))

    plt.show()
    return


if __name__ == '__main__':
    visualize()