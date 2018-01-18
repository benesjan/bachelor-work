import numpy as np

from create_data_paragraphs import build_topics_paragraphs_index_map
from custom_imports import config
from custom_imports.utils import load_pickle, create_dir, save_sparse_csr


def line_map_to_y(lines):
    """
    This method transforms article ranges to article changes
    :param lines: list of ranges, [article1_start, article1_end, article2_start, article2_end, ...]
    :return: vector with binary values, where 1 is on position of the topic (article) change
    """
    # dimension of y_ is the index of last article line -1 (-1 because there can be n-1 topic changes,
    # where n is number of rows in the prediction matrix)
    dim = line_map[-1] - 1
    y_ = np.zeros((dim, 1))
    for i in range(1, len(lines) - 1, 2):
        y_[lines[i] - 1] = 1
    return y_


def get_next_data(data_names):
    for name in data_names:
        yield config.get_seg_data(name)


if __name__ == '__main__':
    print('Loading the instances')
    classifier = load_pickle(config.classifier)
    vectorizer = load_pickle(config.vectorizer)

    for data in get_next_data(['held_out', 'test']):
        print('Processing ' + data['name'] + ' data')

        create_dir(data['dir'])

        articles, ignored, line_map = build_topics_paragraphs_index_map(data['text'])

        y_true = line_map_to_y(line_map)

        print('Building the data matrix using the TfidfVectorizer')
        x = vectorizer.transform(articles)

        del articles, ignored, line_map

        print('Classifying...')
        y = classifier.predict_proba(x)

        print('Saving the data')
        save_sparse_csr(data['x'], x)
        np.save(data['y'], y)
        np.save(data['y_true'], y_true)
