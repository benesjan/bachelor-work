import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from custom_imports import config
from custom_imports.utils import load_pickle, load_sparse_csr, save_pickle


def threshold_half_max(y):
    """
    This method finds the biggest probability and uses half of this probability as threshold.
    :param y: predictions to threshold
    :return: processed predictions
    """
    max_indices = np.argmax(y, axis=1)
    threshold_array = y[np.arange(y.shape[0]), max_indices] / 2

    for i in range(max_indices.shape[0]):
        y[i, :] = y[i, :] > threshold_array[i]

    return y


def get_next(line_map):
    """
    :param line_map: a list with indexes of beginnings and ends of articles
    :return: first line index, last line index and the index of article
    """
    for j in range(len(line_map)):
        if j % 2 == 0:
            yield line_map[j], line_map[j + 1], j / 2


if __name__ == '__main__':
    print("Loading the data")
    y = np.load(config.y_par)
    y_true = np.load(config.y_par_true)
    line_map = load_pickle(config.line_map)

    print("Processing the predictions")
    for line_start, line_end, article_index in get_next(line_map):
        # set all the topics which were not in the original article to 0
        y_article = \
            y_true[article_index] * y[line_start:line_end, :]

        y[line_start:line_end, :] = threshold_half_max(y_article)

    # Check if every topic was used at least once
    if 0 in np.sum(y, axis=0):
        print("WARNING: not all topics used")

    print("Loading x")
    x = load_sparse_csr(config.x_par)

    classifier_one_class = CalibratedClassifierCV(LinearSVC(), cv=3)
    classifier = OneVsRestClassifier(classifier_one_class, n_jobs=1)

    print("Training the classifier")
    classifier.fit(x, y)

    print("Saving the classifier to file")
    save_pickle(config.classifier_par, classifier)
