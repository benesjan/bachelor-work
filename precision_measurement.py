# coding: utf-8
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs

import config
from data_processor import build_corpus_and_topics
from data_utils import load_pickle


# Returns how many articles were not labeled
# def calculate_rows_without_topic(y_pred):
#     with_predictions = 0
#     for row in y_pred:
#         for x in row:
#             if x != 0:
#                 with_predictions += 1
#                 break
#     return y_pred.shape[0] - with_predictions


def r_cut(y, rank=3):
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
    x = vectorizer.transform(corpus)
    print("Transforming article topics by binarizer")
    y_true = binarizer.transform(topics)

    print("Classifying the data")
    y_pred_raw = classifier.decision_function(x)

    # Ensures at least 2 predicted topics for each article
    # 2 topics are optimal for max F on testing data
    y_pred_min_topics = r_cut(y_true, 2)

    # Returns matrix where each elements is set to True if the element's value is bigger than threshold
    y_pred_T = y_pred_raw > 0.16  # -0.14 for 1 min topic

    y_pred = y_pred_min_topics + y_pred_T

    P, R, F, S = prfs(y_true, y_pred, average="samples")
    print('F1 = %.3f (P = %.3f, R = %.3f)' % (F, P, R))
