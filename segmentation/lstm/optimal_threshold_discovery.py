from keras.models import load_model
import numpy as np

from segmentation.lstm.model_training import change_data_ratio
from utils import plot_thresholds
import config

if __name__ == '__main__':
    [X_train, y_train, X_test, y_test] = change_data_ratio(config.get_seg_data('held_out'), config.get_seg_data('test'),
                                                           ratio=0.7)

    del X_train, y_train

    # Add dimension to 2D data
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Append 0 value to y, to get the same number of samples for X and y
    y_test = np.append(y_test, 0)

    model = load_model(config.lstm_model)

    y_pred = model.predict(X_test)

    plot_thresholds(y_test, y_pred, False, 'binary')