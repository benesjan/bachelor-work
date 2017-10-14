# coding: utf-8
import config
import numpy as np
from data_utils import load_pickle
from data_processor import build_corpus_and_topics
from sklearn.metrics import precision_recall_fscore_support as prfs


def RCut(y, rank=3):
    """Apply rank-based thresholding on given matrix.

    In RCut (also known as `k-per-doc`), only `rank` best topics are assigned to
    each document.

    """
    y = np.array(y)
    y_pred = np.zeros(y.shape, dtype=bool)
    for i, row in enumerate(y):
        max_js = row.argsort()[-rank:][::-1]
        for j in max_js:
            y_pred[i, j] = True
    return y_pred


if __name__ == '__main__':
    classifier = load_pickle(config.classifier_path)
    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)

    corpus, topics = build_corpus_and_topics(config.testing_data_path)

    print("Transforming corpus by vectorizer")
    X = vectorizer.transform(corpus)
    print("Transforming article topics by binarizer")
    Y_true = binarizer.transform(topics)

    print("Classifying the data")
    Y_pred = classifier.predict(X)

    Y_pred_bin = RCut(Y_pred)
    Y_true_bin = RCut(Y_true)

    P, R, F, S = prfs(Y_true, Y_pred_bin, average="samples")
    print('F1 = %.3f (P = %.3f, R = %.3f)' % (F, P, R))
