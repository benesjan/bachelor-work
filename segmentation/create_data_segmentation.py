import numpy as np

import config
from utils import load_pickle, create_dir


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


def get_next_data(data_names):
    for name in data_names:
        yield config.get_seg_data(name)


if __name__ == '__main__':
    print('Loading the instances')
    classifier = load_pickle(config.classifier)

    for data in get_next_data(config.data.keys()):
        print('Processing ' + data['name'] + ' data')

        create_dir(data['dir'])

        y_true = line_map_to_y(load_pickle(data['line_map']))

        print('Saving the data')
        np.save(data['y_true_lm'], y_true)
