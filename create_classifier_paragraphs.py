import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from custom_imports import config
from custom_imports.utils import load_pickle, load_sparse_csr, save_pickle, choose_option


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


def threshold_biggest_gap(y):
    """
    This method finds the biggest gap between sorted predictions and uses position of this gap as threshold.
    :param y: predictions to threshold
    :return: processed predictions
    """
    y_sorted = -np.sort(-y, axis=1)

    for i in range(y.shape[0]):
        biggest_gap = 0
        gap_index = -1
        for j in range(y.shape[1] - 1):
            gap = y_sorted[i, j] - y_sorted[i, j + 1]
            if gap > biggest_gap:
                biggest_gap = gap
                gap_index = j

            # If the element is smaller than the biggest gap then there can't be bigger gap found in the row
            if biggest_gap > y[i, j + 1]:
                break

        assert biggest_gap != 0 and gap_index != -1, "Variables not set"

        # Probability between the 2 furthest probabilities is set as threshold
        threshold = (y_sorted[i, gap_index] + y_sorted[i, gap_index + 1]) / 2

        # Apply the threshold to row
        y[i, :] = y[i, :] > threshold

    return y


def get_next(line_map):
    """
    :param line_map: a list with indexes of beginnings and ends of articles
    :return: first line index, last line index and the index of article
    """
    for j in range(len(line_map)):
        if j % 2 == 0:
            yield line_map[j], line_map[j + 1], j / 2


def process_y(data, func):
    """
    This  method removes the topic predictions which were not in the full article from paragraph predictions
    :param data: paths to data
    :param func: a function that will be applied to paragraphs
    :return: y from which non article topics were removed and to which threshold function was applied
    """
    print("Loading the data")
    y = np.load(data['y'])
    y_true = np.load(data['y_true'])
    line_map = load_pickle(data['line_map'])

    print("Processing the predictions")
    for line_start, line_end, article_index in get_next(line_map):
        # set all the topics which were not in the original article to 0
        y_article = y_true[article_index] * y[line_start:line_end, :]

        y[line_start:line_end, :] = func(y_article)

    return y


if __name__ == '__main__':
    data = config.get_par_data('train')

    if choose_option('Do you want to use biggest gap thresholding mechanism [b]' +
                     ' or half the biggest probability as threshold [h]?', 'b', 'h'):
        threshold_func = threshold_biggest_gap
        classifier_path = config.classifier_par_biggest_gap
    else:
        threshold_func = threshold_half_max
        classifier_path = config.classifier_par_half_max

    y = process_y(data, threshold_func)

    # Check if every topic was used at least once
    if 0 in np.sum(y, axis=0):
        print("WARNING: not all topics used")

    print("Loading x")
    x = load_sparse_csr(data['x'])

    classifier_one_class = CalibratedClassifierCV(LinearSVC(), cv=3)
    classifier = OneVsRestClassifier(classifier_one_class, n_jobs=1)

    print("Training the classifier")
    classifier.fit(x, y)

    print("Saving the classifier to file")
    save_pickle(classifier_path, classifier)
