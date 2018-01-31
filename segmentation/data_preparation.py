import numpy as np

import config
from utils import load_pickle, create_dir, save_sparse_csr, build_topics_paragraphs_index_map, load_sparse_csr


def line_map_to_y(lines):
    """
    This method transforms article ranges to article changes
    :param lines: list of ranges, [article1_start, article1_end, article2_start, article2_end, ...]
    :return: vector with binary values, where 1 is on position of the topic (article) change
    """
    # dimension of y_ is the index of last article line -1 (-1 because there can be n-1 topic changes,
    # where n is number of rows in the prediction matrix)
    dim = lines[-1] - 1
    y_ = np.zeros((dim, 1))
    for i in range(1, len(lines) - 1, 2):
        y_[lines[i] - 1] = 1
    return y_


if __name__ == '__main__':
    print('Loading the instances')
    classifier = load_pickle(config.classifier)

    for data_name in ['held_out', 'test']:
        print('Processing ' + data_name + ' data')

        data_seg = config.get_seg_data(data_name)
        data_par = config.get_par_data(data_name)

        create_dir(data_seg['dir'])

        y_true = line_map_to_y(load_pickle(data_par['line_map']))

        print("Loading x")
        x = load_sparse_csr(data_par['x'])

        print('Classifying...')
        y = classifier.predict_proba(x)

        print('Saving the data')
        save_sparse_csr(data_seg['x'], x)
        np.save(data_seg['y'], y)
        np.save(data_seg['y_true'], y_true)
