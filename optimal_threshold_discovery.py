# coding: utf-8
import numpy as np
from matplotlib import pyplot
from sklearn.metrics import precision_recall_fscore_support as prfs

from custom_imports import config
from custom_imports.utils import load_pickle, build_corpus_and_topics, r_cut

if __name__ == '__main__':
    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)
    classifier = load_pickle(config.classifier_path)

    corpus, topics = build_corpus_and_topics(config.held_out_data_path)

    print("Transforming corpus by vectorizer")
    x = vectorizer.transform(corpus)
    print("Transforming article topics by binarizer")
    y_true = binarizer.transform(topics)

    print("Classifying the data")
    y_dec = classifier.decision_function(x)

    # Ensures at least 1 predicted topic for each article
    y_pred_min_topics = r_cut(y_dec, 1)

    threshold_array = np.arange(-1.0, 1.0, 0.01)

    values = np.ones((threshold_array.shape[0], 4), dtype=np.float)

    for i, T in enumerate(threshold_array):
        # Returns matrix where each elements is set to True if the element's value is bigger than threshold
        y_pred_T = y_dec > T

        y_pred = y_pred_min_topics + y_pred_T

        P, R, F, S = prfs(y_true, y_pred, average="samples")
        values[i, :] = [T, F, P, R]

        print('threshold = %.2f, F1 = %.3f (P = %.3f, R = %.3f)' % (T, F, P, R))

    pyplot.plot(values[:, 0], values[:, 1:4])
    pyplot.legend(['F-measure', 'Precision', 'Recall'])

    pyplot.title('Hledání optimálního prahu')
    pyplot.xlabel('Práh')
    pyplot.show()
