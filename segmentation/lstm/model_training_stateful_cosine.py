from pathlib import Path

import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_distances

import config
from segmentation.distance_based_methods import compute_distance
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

    return [x[0:num_train_rows], y[0:num_train_rows - 1], x[num_train_rows:, :], y[num_train_rows:]]


def build_model():
    _model = Sequential()

    # stateful=True means that the state is propagated to the next batch
    _model.add(LSTM(32, batch_input_shape=(1, 1, 1), stateful=True))
    # _model.add(Dropout(0.5))
    _model.add(Dense(1, activation='sigmoid'))
    _model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return _model


if __name__ == '__main__':
    path = Path(config.lstm_model)
    if path.is_file():
        if first_option('Do you want to continue training the saved model?', 'y', 'n'):
            print("Loading new model")
            model = load_model(config.lstm_model)
        else:
            print("Renaming the model")
            path.rename(config.seg_data_dir + "/lstm_model_old.h5")
            model = build_model()
    else:
        model = build_model()

    print("Processing the data")
    [X_train, y_train, X_test, y_test] = change_data_ratio(config.get_seg_data('held_out'), config.get_seg_data('test'),
                                                           ratio=0.7)
    np.random.seed(7)

    print("Computing the distances")
    X_train = compute_distance(X_train, cosine_distances)
    X_test = compute_distance(X_test, cosine_distances)

    # Add dimension to 2D data
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    epochs = 5
    for i in range(epochs):
        print("Epoch " + str(i) + "/" + str(epochs) + ":")
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=1, shuffle=False)
        model.reset_states()

    model.save(config.lstm_model)

    print(model.summary())
