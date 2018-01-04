# coding: utf-8
import numpy as np
from matplotlib import pyplot, rc
from sklearn.metrics import precision_recall_fscore_support as prfs

from custom_imports import config
from custom_imports.utils import load_pickle, build_corpus_and_topics, r_cut

if __name__ == '__main__':
    vectorizer = load_pickle(config.data_vectorizer_path)
    binarizer = load_pickle(config.topic_binarizer_path)

    print('Loading calibrated classifier')
    classifier = load_pickle(config.classifier_path)

    corpus, topics = build_corpus_and_topics(config.held_out_data_path)

    print("Transforming corpus by vectorizer")
    x = vectorizer.transform(corpus)
    print("Transforming article topics by binarizer")
    y_true = binarizer.transform(topics)

    print("Classifying the data")
    y_pred = classifier.predict_proba(x)

    threshold_array = np.arange(0, 1.0, 0.01)

    # Ensures at least 1 predicted topic for each article
    y_pred_min_topics = r_cut(y_pred, 1)

    values = np.ones((threshold_array.shape[0], 4), dtype=np.float)

    optimal_position = [-100, 0, 0, 0]

    for i, T in enumerate(threshold_array):
        # Returns matrix where each elements is set to True if the element's value is bigger than threshold
        y_pred_T = y_pred > T

        y_classifier = y_pred_min_topics + y_pred_T

        P, R, F, S = prfs(y_true, y_classifier, average="samples")
        values[i, :] = [T, F, P, R]

        print('threshold = %.2f, F1 = %.3f (P = %.3f, R = %.3f)' % (T, F, P, R))

        if F > optimal_position[1]:
            optimal_position = values[i, :]

    rc('font', family='Arial')
    pyplot.plot(values[:, 0], values[:, 1:4])
    pyplot.legend(['F-measure', 'Precision', 'Recall'])

    pyplot.grid()
    pyplot.scatter(optimal_position[0], optimal_position[1], marker="x", s=300, linewidth=3.3)
    pyplot.annotate('[%.2f, %.2f]' % (optimal_position[0], optimal_position[1]),
                    [optimal_position[0], optimal_position[1] - 0.055])

    pyplot.xlim([0, 1])
    pyplot.ylim([0, 1])

    pyplot.title('Vývoj přesnosti, úplnosti a F-míry v závislosti na prahu')
    pyplot.xlabel('Práh')

    pyplot.show()
