# coding: utf-8
import pickle
from pathlib import Path
from nltk import windowdiff, pk
from scipy.sparse import csr_matrix
import numpy as np
from matplotlib import pyplot, rc
from sklearn.metrics import precision_recall_fscore_support as prfs
from re import match


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


def build_corpus_and_topics(data_file_path, n_articles=-1):
    """
    :param data_file_path: path to the data file
    :param n_articles: process only first n articles, if left blank all are processed
    :return: data and topics lists in a not-vectorized form
    """
    pattern = r'<article id="([0-9]+)" topics="(.*)">'
    corpus, topics = [], []
    current_article = ""

    articles_processed = 0
    with open(data_file_path, 'r', encoding='utf-8') as handler:
        for line in handler:
            if line.startswith('<'):
                if line == '</article>\n':
                    corpus.append(current_article)
                    if articles_processed == n_articles:
                        break
                    current_article = ""
                    continue

                match_obj = match(pattern, line)
                if match_obj:
                    topics.append(match_obj.group(2).split(' '))
                    articles_processed += 1
                    if current_article:
                        print("Warning: Non empty article string")
                    print(str(articles_processed) + ". article loaded")
                    continue

            current_article += line

    if len(corpus) != len(topics):
        raise ValueError("Error: matrix dimensions do not match")

    return corpus, topics


def build_topics_paragraphs_index_map(data_path, n_articles=-1):
    pattern = r'<article id="([0-9]+)" topics="(.*)">'

    articles, topics, line_map = [], [], []
    articles_processed, paragraph_index = 0, 0

    with open(data_path, 'r', encoding='utf-8') as handler:
        for line in handler:
            if line.startswith('<'):
                if line == '</article>\n':
                    line_map.append(paragraph_index)
                    # number of elements in line_map should be even since it contains start-end pairs
                    assert len(line_map) % 2 == 0
                    # End of article
                    if articles_processed == n_articles:
                        break
                    continue

                match_obj = match(pattern, line)
                if match_obj:
                    line_map.append(paragraph_index)
                    assert len(line_map) % 2 == 1
                    # First line of article
                    topics.append(match_obj.group(2).split(' '))
                    articles_processed += 1
                    print(str(articles_processed) + '. article loaded')
                    continue

            paragraph_index += 1
            articles.append(line)

    if len(line_map) != (len(topics) * 2):
        raise ValueError('Error: matrix dimensions do not match')

    return articles, topics, line_map


def save_pickle(file_path, object_to_save):
    with open(file_path, 'wb') as handle:
        pickle.dump(object_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def first_option(question, first, second):
    while True:
        reply = str(input(question + ' ({0}/{1}): '.format(first, second))).lower().strip()
        if reply[0] == first:
            return True
        if reply[0] == second:
            return False
        print("Incorrect input, please enter '{0}' or '{1}'".format(first, second))


def plot_thresholds(y_true, y_pred, ensure_topic=False, average_type='samples', interval=(0, 1)):
    threshold_array = np.arange(interval[0], interval[1], 0.01)

    values = np.ones((threshold_array.shape[0], 4), dtype=np.float)

    if ensure_topic:
        # Ensures at least 1 predicted topic for each article
        y_pred_min_topics = r_cut(y_pred, 1)
    else:
        y_pred_min_topics = 0

    optimal_position = [-100, 0, 0, 0]

    for i, T in enumerate(threshold_array):
        y_classifier = y_pred_min_topics + (y_pred > T)

        P, R, F, S = prfs(y_true, y_classifier, average=average_type)
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

    pyplot.xlim(interval)
    pyplot.ylim([0, 1])

    pyplot.title('Vývoj přesnosti, úplnosti a F-míry v závislosti na prahu')
    pyplot.xlabel('Práh')

    pyplot.show()


def print_measurements(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    P, R, F, S = prfs(y_true, y_pred, average='binary')

    window_length = 4  # average segment length is 7.22 --> half is ~ 4

    y_true_joined = "".join(["" + str(int(x)) for x in y_true])
    y_pred_joined = "".join(["" + "1" if x else "0" for x in y_pred])
    wd = windowdiff(y_true_joined, y_pred_joined, k=window_length, boundary="1")
    pk_m = pk(y_true_joined, y_pred_joined, k=window_length, boundary="1")

    print('F1 = {0:.3f} (P = {1:.3f}, R = {2:.3f}), WindowDiff = {3:.3f}, Pk = {4:.3f}'.format(F, P, R, wd, pk_m))
