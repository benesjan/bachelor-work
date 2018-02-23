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
    x_tr = np.load(train['y'])
    y_tr = np.load(train['y_true_lm'])

    x_te = np.load(test['y'])
    y_te = np.load(test['y_true_lm'])

    x = np.concatenate((x_tr, x_te), axis=0)
    y = np.concatenate((y_tr, np.ones((1, 1)), y_te))

    num_of_train_rows = int(x.shape[0] * 0.7)

    return [x[0:num_of_train_rows], y[0:num_of_train_rows - 1], x[num_of_train_rows:, :], y[num_of_train_rows:]]


def process_data(x, steps=200):
    # Remove the last few training samples so that every time step consists of "steps" vectors
    n = int(len(x) / steps) * steps
    x_list = list()
    for i in range(0, n, steps):
        sample = x[i:i + steps]
        x_list.append(sample)
    return np.array(x_list)


if __name__ == '__main__':
    # If true, stateful LSTM will be used and batch shuffling will be disabled
    stateful = False

    # Number of vectors in one sequence, input data structure [samples, time_steps, features]
    time_steps = 200

    [X_train, y_train, X_test, y_test] = change_data_ratio(config.get_seg_data('held_out'), config.get_seg_data('test'),
                                                           ratio=0.7)
    np.random.seed(7)

    # Split the 2D matrix to 3D matrix of dimensions [samples, time_steps, features]
    X_train = process_data(X_train)
    X_test = process_data(X_test)

    # Append 0 value to the beginning of y so the values represent if there was a boundary between current sample and
    # the previous one, not the current and next
    y_train = np.append(0, y_train)
    y_test = np.append(0, y_test)

    # Split the 1D vector to 2D matrix of dimensions: [samples, time_steps]
    y_train = process_data(y_train)
    y_test = process_data(y_test)

    model = Sequential()

    # stateful=True means that the state is propagated to the next batch
    model.add(LSTM(100, input_shape=(time_steps, 577), stateful=stateful))

    model.add(Dense(time_steps, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=1, shuffle=(not stateful))

    model.save(config.lstm_model)

    print(model.summary())
