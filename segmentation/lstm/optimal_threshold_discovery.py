from keras.models import load_model
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

import config
from segmentation.distance_based_methods import compute_distance
from segmentation.lstm.lstm_utils import get_data, split_to_time_steps
from utils import first_option, plot_thresholds

if __name__ == '__main__':
    if first_option('Do you want to use the model trained on cosine distances [c] or on raw SVM predictions [r]?',
                    'c', 'r'):
        cosine = True
        model = load_model(config.lstm_model_1)
    else:
        cosine = False
        model = load_model(config.lstm_model_577)

    time_steps = model.get_config()[0]['config']['batch_input_shape'][1]

    [X_train, y_train, X_held, y_held, X_test, y_test] = get_data(time_steps)

    del X_train, y_train, X_test, y_test

    if cosine:
        print("Computing the distances")
        X_held = compute_distance(X_held, cosine_distances)
    else:
        y_held = np.append(0, y_held)

    X = split_to_time_steps(X_held)
    y_true = split_to_time_steps(y_held)

    y_pred = model.predict(X)

    plot_thresholds(y_true.flatten(), y_pred.flatten(), False, average_type="binary")
