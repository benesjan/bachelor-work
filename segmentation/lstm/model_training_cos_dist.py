from pathlib import Path

import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_distances

import config
from segmentation.distance_based_methods import compute_distance
from segmentation.lstm.lstm_utils import split_to_time_steps, get_data
from utils import first_option, plot_thresholds


def build_model(_time_steps=200):
    _model = Sequential()

    # stateful=True means that the state is propagated to the next batch
    _model.add(LSTM(32, input_shape=(_time_steps, 1), stateful=False, return_sequences=True))
    _model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    _model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return _model


if __name__ == '__main__':
    # Number of vectors in one sequence, input data structure [samples, time_steps, features]
    time_steps = 200

    np.random.seed(7)

    if Path(config.lstm_model_1).is_file() and first_option('Do you want to continue training the saved model?',
                                                            'y', 'n'):
        print("Loading new model")
        model = load_model(config.lstm_model_1)
    else:
        print("Building new model")
        model = build_model(time_steps)

    print("Processing the data")
    [X_train, y_train, X_held, y_held, X_test, y_test] = get_data(time_steps)

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

    model.save(config.lstm_model_1)

    print(model.summary())

    y_pred = model.predict(X_test)

    plot_thresholds(y_test, y_pred, ensure_topic=False, average_type='samples')
