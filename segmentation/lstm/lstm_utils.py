import numpy as np

import config


def _split_in_half(x, y, steps):
    """
    Split the data in half
    :param steps: number of steps/vectors in time sequence
    :param x: data to split
    :param y: labels to split
    :return: [X_1st_half, y_1st_half, X_2nd_hald, y_2nd_half]
    """

    num_train_rows = int(x.shape[0] * 0.5)

    # Minimizes the data loss
    num_train_rows = int(num_train_rows / steps) * steps

    return [x[0:num_train_rows], y[0:num_train_rows - 1], x[num_train_rows:, :], y[num_train_rows:]]


def get_data(steps):
    """
    Split the data to train, test and held out data
    :param steps: number of steps/vectors in time sequence
    :return: train, test and held out data
    """

    held_out = config.get_seg_data('held_out')
    test = config.get_seg_data('test')

    x_ho = np.load(held_out['y'])
    ho = np.load(held_out['y_true_lm'])

    x_te = np.load(test['y'])
    y_te = np.load(test['y_true_lm'])

    x = np.concatenate((x_ho, x_te), axis=0)
    y = np.concatenate((ho, np.ones((1, 1)), y_te))

    [X_train, y_train, X_rest, y_rest] = _split_in_half(x, y, steps)
    [X_held, y_held, X_test, y_test] = _split_in_half(X_rest, y_rest, steps)

    return [X_train, y_train, X_held, y_held, X_test, y_test]


def split_to_time_steps(x, steps=200):
    """
    Processes the data so that they are suitable for LSTMs
    :param x: 2D matrix
    :param steps: number of steps in the time sequence
    :return: 3D matrix with the following dimensions: [samples, time_steps, features]
    """
    n = int(len(x) / steps) * steps
    x_list = list()
    for i in range(0, n, steps):
        sample = x[i:i + steps]
        x_list.append(sample)
    return np.array(x_list)
