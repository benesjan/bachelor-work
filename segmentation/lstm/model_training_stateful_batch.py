from pathlib import Path

import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model

import config
from utils import first_option


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

    x_tr = np.load(train['y'])
    y_tr = np.load(train['y_true_lm'])

    x_te = np.load(test['y'])
    y_te = np.load(test['y_true_lm'])

    x = np.concatenate((x_tr, x_te), axis=0)
    y = np.concatenate((y_tr, np.ones((1, 1)), y_te))

    num_train_rows = int(x.shape[0] * 0.7)

    # Minimizes data loss
    num_train_rows = int(num_train_rows / 100) * 100

    return [x[0:num_train_rows], y[0:num_train_rows - 1], x[num_train_rows:, :], y[num_train_rows:]]


def create_batches(x, batch_size=100):
    ax_1_length = int(x.shape[0] / batch_size) * batch_size
    x = x[0:ax_1_length]

    x_processed = np.zeros(x.shape)
    x_counter = 0

    for position_within_batch in range(0, batch_size):
        for batch_start in range(0, ax_1_length, 100):
            assert np.amax(x_processed[batch_start + position_within_batch]) == 0, \
                "Bug in create_batches, batch_start = " + str(batch_start) + ", position_within_batch = " \
                + str(position_within_batch)
            x_processed[batch_start + position_within_batch] = x[x_counter]
            x_counter += 1
    return x_processed


def build_model(batch_size):
    _model = Sequential()

    # stateful=True means that the state is propagated to the next batch
    _model.add(LSTM(512, batch_input_shape=(batch_size, 1, 577), stateful=True))
    # _model.add(Dropout(0.5))
    _model.add(Dense(1, activation='sigmoid'))
    _model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return _model


if __name__ == '__main__':
    batch_size = 100
    path = Path(config.lstm_model)
    if path.is_file():
        if first_option('Do you want to continue training the saved model?', 'y', 'n'):
            print("Loading new model")
            model = load_model(config.lstm_model)
        else:
            print("Renaming the model")
            path.rename(config.seg_data_dir + "/lstm_model_old.h5")
            model = build_model(batch_size)
    else:
        model = build_model(batch_size)

    print("Processing the data")
    [X_train, y_train, X_test, y_test] = change_data_ratio(config.get_seg_data('held_out'), config.get_seg_data('test'),
                                                           ratio=0.7)
    np.random.seed(7)

    # Append 0 value to the beginning of y so the values represent if there was a boundary between current sample and
    # the previous one, not the current and next
    y_train = np.append(0, y_train)
    y_test = np.append(0, y_test)

    X_train = create_batches(X_train, batch_size=batch_size)
    y_train = create_batches(y_train, batch_size=batch_size)
    X_test = create_batches(X_test, batch_size=batch_size)
    y_test = create_batches(y_test, batch_size=batch_size)

    # Add dimension to 2D data
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    for i in range(100):
        print("Epoch " + str(i) + ":")
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=batch_size, shuffle=False)
        model.reset_states()

    model.save(config.lstm_model)

    print(model.summary())
