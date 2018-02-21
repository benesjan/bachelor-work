import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense

import config


def change_data_ratio(train, test, ratio=0.7):
    """
    Split data according to new ratio
    :param train: path to train data
    :param test: path to test data
    :param ratio: ratio to split data by
    :return: [X_train, y_train, X_test, y_test] split according to ratio
    """
    if ratio < 0.1 or ratio > 0.95:
        raise ValueError("Invalid ratio value: " + str(ratio))

    print("Loading the predictions and thematic boundaries")
    x_train = np.load(train['y'])
    y_train = np.load(train['y_true_lm'])

    x_test = np.load(test['y'])
    y_test = np.load(test['y_true_lm'])

    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, np.ones((1, 1)), y_test))

    num_of_train_rows = int(x.shape[0] * 0.7)

    return [x[0:num_of_train_rows], y[0:num_of_train_rows - 1], x[num_of_train_rows:, :], y[num_of_train_rows:]]


if __name__ == '__main__':
    [X_train, y_train, X_test, y_test] = change_data_ratio(config.get_seg_data('held_out'), config.get_seg_data('test'),
                                                           ratio=0.7)
    np.random.seed(7)

    # Add dimension to 2D data
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Append 0 value to y, to get the same number of samples for X and y
    y_train = np.append(y_train, 0)
    y_test = np.append(y_test, 0)

    model = Sequential()
    model.add(LSTM(100, input_shape=(1, 577)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # batch_size=500000 in order to propagate all the data at once
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=500000)

    model.save(config.lstm_model)