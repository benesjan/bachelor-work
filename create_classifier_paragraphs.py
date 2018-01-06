import numpy as np
from custom_imports import config
from custom_imports.utils import load_pickle


def threshold_half_max(y):
    max_indices = np.argmax(y, axis=1)
    threshold_array = y[np.arange(y.shape[0]), max_indices] / 2

    for i in range(max_indices.shape[0]):
        y[i, :] = y[i, :] > threshold_array[i]

    return y


def get_next(article_map_):
    for j in range(len(article_map_)):
        if j % 2 == 0:
            yield article_map_[j], article_map_[j + 1], j / 2


if __name__ == '__main__':
    y_pred = np.load(config.y_paragraphs)
    y_true = np.load(config.y_paragraphs_true)

    article_map = load_pickle(config.article_paragraph_map)

    for line_start, line_end, article_index in get_next(article_map):
        # set all the topics which were not in the original article to 0
        y_article = \
            y_true[article_index] * y_pred[line_start:line_end, :]

        y_pred[line_start:line_end, :] = threshold_half_max(y_article)
