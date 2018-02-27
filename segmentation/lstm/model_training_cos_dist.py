from pathlib import Path
import numpy as np
from keras.models import load_model
from sklearn.metrics.pairwise import cosine_distances

import config
from segmentation.distance_based_methods import compute_distance
from segmentation.lstm.lstm_utils import split_to_time_steps, get_data, shuffle_the_data, build_model
from utils import first_option

if __name__ == '__main__':
    # Number of vectors in one sequence, input data structure [samples, time_steps, features]
    time_steps = 200

    if Path(config.lstm_model_1).is_file() and first_option('Do you want to continue training the saved model?',
                                                            'y', 'n'):
        print("Loading new model")
        model = load_model(config.lstm_model_1)
    else:
        print("Building new model")
        model = build_model(time_steps, 1)

    print("Processing the data")
    [X_train_or, y_train_or, X_held, y_held, X_test, y_test] = get_data(time_steps)

    del X_held, y_held

    print("Computing the distances on test data")
    X_test = compute_distance(X_test, cosine_distances)

    # Split the 2D matrix to 3D matrix of dimensions [samples, time_steps, features]
    X_test = split_to_time_steps(X_test)

    # Split the 1D vector to 2D matrix of dimensions: [samples, time_steps]
    y_test = split_to_time_steps(y_test)

    y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1], 1))

    # Append 0 value to the beginning of y so the values represent if there was a boundary between current sample and
    # the previous one, not the current and next
    y_train_or = np.append(0, y_train_or)
    shuffling_epochs = 5
    for i in range(shuffling_epochs):
        print("Shuffling epoch " + str(i) + "/" + str(shuffling_epochs))

        [X_train, y_train] = shuffle_the_data(X_train_or, y_train_or)

        print("Computing the distances on training data")
        X_train = compute_distance(X_train, cosine_distances)

        # 0 padding is required in this case even for X matrix, because the matrix consists of distances and we need
        # the distance to be between current sample and the previous one, not the current and next
        X_train = np.append(0, X_train)

        X_train = split_to_time_steps(X_train)
        y_train = split_to_time_steps(y_train)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=100, shuffle=False)

    model.save(config.lstm_model_1)
