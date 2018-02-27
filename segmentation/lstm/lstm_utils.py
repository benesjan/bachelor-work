import numpy as np
from keras import Sequential
from keras.layers import LSTM, TimeDistributed, Dense

import config


def _split_in_half(X, y, steps):
    """
    Split the data in half
    :param steps: number of steps/vectors in time sequence
    :param X: data to split
    :param y: labels to split
    :return: [X_1st_half, y_1st_half, X_2nd_hald, y_2nd_half]
    """

    num_train_rows = int(X.shape[0] * 0.5)

    # Minimizes the data loss
    num_train_rows = int(num_train_rows / steps) * steps

    return [X[0:num_train_rows], y[0:num_train_rows - 1], X[num_train_rows:, :], y[num_train_rows:]]


def get_data(steps):
    """
    Split the data to train, test and held out data
    :param steps: number of steps/vectors in time sequence
    :return: train, test and held out data
    """

    held_out = config.get_seg_data('held_out')
    test = config.get_seg_data('test')

    X_ho = np.load(held_out['y'])
    y_ho = np.load(held_out['y_true_lm'])

    X_te = np.load(test['y'])
    y_te = np.load(test['y_true_lm'])

    X = np.concatenate((X_ho, X_te), axis=0)
    y = np.concatenate((y_ho, np.ones((1, 1)), y_te))

    [X_train, y_train, X_rest, y_rest] = _split_in_half(X, y, steps)
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


def shuffle_the_data(X, y):
    y_shuffled = np.zeros(y.shape, dtype=int)

    # Ensures that the last article is not omitted
    y = np.append(y, 1)
    X_articles = []
    article_start = 0
    for i, val in enumerate(y):
        if val == 1:
            X_articles.append(X[article_start:i])
            article_start = i

    np.random.shuffle(X_articles)

    article_end = 0
    for article in X_articles:
        article_end += len(article)
        if article_end != y_shuffled.shape[0]:
            y_shuffled[article_end] = 1

    X_shuffled = np.vstack(X_articles)
    return [X_shuffled, y_shuffled]


def build_model(time_steps, n_features):
    model = Sequential()

    # stateful=True means that the state is propagated to the next batch
    model.add(LSTM(32, input_shape=(time_steps, n_features), stateful=False, return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
