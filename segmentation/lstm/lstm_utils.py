import numpy as np


def change_data_ratio(train, test, ratio=0.7, steps=200):
    """
    Split data according to new ratio
    :param steps: number of steps/vectors in time sequence
    :param train: path to train data
    :param test: path to test data
    :param ratio: ratio to split data by
    :return: [X_train, y_train, X_test, y_test] split according to ratio
    """
    if ratio < 0.1 or ratio > 0.95:
        raise ValueError("Invalid ratio value: " + str(ratio))

    x_tr = np.load(train['y'])
    y_tr = np.load(train['y_true_lm'])

    x_te = np.load(test['y'])
    y_te = np.load(test['y_true_lm'])

    x = np.concatenate((x_tr, x_te), axis=0)
    y = np.concatenate((y_tr, np.ones((1, 1)), y_te))

    num_train_rows = int(x.shape[0] * 0.7)

    # Minimizes data loss
    num_train_rows = int(num_train_rows / steps) * steps

    return [x[0:num_train_rows], y[0:num_train_rows - 1], x[num_train_rows:, :], y[num_train_rows:]]


def split_to_time_steps(x, steps=200):
    # Remove the last few training samples so that every time step consists of "steps" vectors
    n = int(len(x) / steps) * steps
    x_list = list()
    for i in range(0, n, steps):
        sample = x[i:i + steps]
        x_list.append(sample)
    return np.array(x_list)
