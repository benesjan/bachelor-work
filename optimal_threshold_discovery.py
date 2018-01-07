# coding: utf-8
import numpy as np
from matplotlib import pyplot, rc
from sklearn.metrics import precision_recall_fscore_support as prfs

from create_classifier_paragraphs import threshold_half_max, process_y
from custom_imports import config
from custom_imports.utils import load_pickle, build_corpus_and_topics, r_cut, load_sparse_csr, choose_option

if __name__ == '__main__':

    if choose_option('Do you want to use paragraphs trained classifier or the  article trained version?', 'p', 'a'):
        data = config.get_par_data('held_out')

        print('Loading the paragraph trained classifier')
        classifier = load_pickle(config.classifier_par)

        y_true = process_y(data, threshold_half_max)

        print("Loading x")
        x = load_sparse_csr(data['x'])
    else:
        print('Loading the full article trained classifier')
        vectorizer = load_pickle(config.vectorizer)
        binarizer = load_pickle(config.binarizer)

        classifier = load_pickle(config.classifier)

        corpus, topics = build_corpus_and_topics(config.data['held_out'])

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
        y_classifier = y_pred_min_topics + (y_pred > T)

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
