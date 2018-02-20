import numpy as np
from matplotlib import pyplot, rc

import config
from utils import build_corpus_and_topics, load_pickle

if __name__ == '__main__':
    vectorizer = load_pickle(config.vectorizer)
    binarizer = load_pickle(config.binarizer)

    corpus, topics = build_corpus_and_topics(config.data['train'])

    print("Building the data matrix using the TfidfVectorizer")
    X = vectorizer.transform(corpus)

    print("Building the label matrix by MultiLabelBinarizer")
    Y = binarizer.transform(topics)

    topic_counts = s = np.sum(Y, axis=1)
    unique, counts = np.unique(s, return_counts=True)

    font = {'family': 'Arial',
            # 'weight': 'bold',
            'size': 16}

    rc('font', **font)

    pyplot.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    pyplot.xlim([0, 11])
    pyplot.xlabel("Počet témat")
    pyplot.ylabel("Počet článků")
    pyplot.bar(unique, counts, color="purple", align='center')

    xi = np.arange(1, len(unique) + 1)
    pyplot.xticks(xi)

    pyplot.show()
