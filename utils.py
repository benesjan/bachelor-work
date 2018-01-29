# coding: utf-8
import pickle
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

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
