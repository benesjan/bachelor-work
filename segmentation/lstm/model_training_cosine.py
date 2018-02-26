import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.models import load_model
from pathlib import Path

from sklearn.metrics.pairwise import cosine_distances

import config
from segmentation.distance_based_methods import compute_distance
from utils import first_option, plot_thresholds


def change_data_ratio(train, test, ratio=0.7, steps=200):
    """
    Split data according to new ratio
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


def build_model(_time_steps=200):
    _model = Sequential()

    # stateful=True means that the state is propagated to the next batch
    _model.add(LSTM(32, input_shape=(_time_steps, 1), stateful=False, return_sequences=True))
    # _model.add(Dropout(0.5))
    _model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    _model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return _model


if __name__ == '__main__':
    # Number of vectors in one sequence, input data structure [samples, time_steps, features]
    time_steps = 200

    np.random.seed(7)

    if Path(config.lstm_model).is_file() and first_option('Do you want to continue training the saved model?',
                                                          'y', 'n'):
        print("Loading new model")
        model = load_model(config.lstm_model)
    else:
        print("Building new model")
        model = build_model(time_steps)

    print("Processing the data")
    [X_train, y_train, X_test, y_test] = change_data_ratio(config.get_seg_data('held_out'), config.get_seg_data('test'),
                                                           ratio=0.7, steps=time_steps)

    print("Computing the distances")
    X_train = compute_distance(X_train, cosine_distances)
    X_test = compute_distance(X_test, cosine_distances)

    # Split the 2D matrix to 3D matrix of dimensions [samples, time_steps, features]
    X_train = split_to_time_steps(X_train)
    X_test = split_to_time_steps(X_test)

    # Split the 1D vector to 2D matrix of dimensions: [samples, time_steps]
    y_train = split_to_time_steps(y_train)
    y_test = split_to_time_steps(y_test)

    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=400, shuffle=True)

    model.save(config.lstm_model)

    print(model.summary())

    y_pred = model.predict(X_test)

    plot_thresholds(y_test, y_pred, ensure_topic=False, average_type='samples')