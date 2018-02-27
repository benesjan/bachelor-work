from pathlib import Path

import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.models import load_model

import config
from segmentation.lstm.lstm_utils import split_to_time_steps, change_data_ratio
from utils import first_option


def build_model(_time_steps=200):
    _model = Sequential()

    # stateful=True means that the state is propagated to the next batch
    _model.add(LSTM(32, input_shape=(_time_steps, 577), stateful=False, return_sequences=True))
    _model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    _model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return _model


if __name__ == '__main__':
    # Number of vectors in one sequence, input data structure [samples, time_steps, features]
    time_steps = 200

    np.random.seed(7)

    if Path(config.lstm_model_577).is_file() and first_option('Do you want to continue training the saved model?',
                                                          'y', 'n'):
        print("Loading new model")
        model = load_model(config.lstm_model_577)
    else:
        print("Building new model")
        model = build_model(time_steps)

    print("Processing the data")
    [X_train, y_train, X_test, y_test] = change_data_ratio(config.get_seg_data('held_out'), config.get_seg_data('test'),
                                                           ratio=0.7, steps=time_steps)

    # Split the 2D matrix to 3D matrix of dimensions [samples, time_steps, features]
    X_train = split_to_time_steps(X_train)
    X_test = split_to_time_steps(X_test)

    # Append 0 value to the beginning of y so the values represent if there was a boundary between current sample and
    # the previous one, not the current and next
    y_train = np.append(0, y_train)
    y_test = np.append(0, y_test)

    # Split the 1D vector to 2D matrix of dimensions: [samples, time_steps]
    y_train = split_to_time_steps(y_train)
    y_test = split_to_time_steps(y_test)

    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=400, shuffle=True)

    model.save(config.lstm_model_577)

    print(model.summary())
