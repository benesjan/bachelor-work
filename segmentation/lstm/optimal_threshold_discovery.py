from keras.models import load_model
import numpy as np

from segmentation.lstm.model_training import change_data_ratio, process_data
from utils import plot_thresholds
import config

if __name__ == '__main__':
    [X_train, y_train, X_test, y_test] = change_data_ratio(config.get_seg_data('held_out'), config.get_seg_data('test'),
                                                           ratio=0.7)

    del X_train, y_train

    # Add dimension to 2D data
    X_test = process_data(X_test)

    y_test = np.append(0, y_test)

    model = load_model(config.lstm_model)

    y_pred = model.predict(X_test)

    y_pred = y_pred.flatten()

    y_test = y_test[0:y_pred.shape[0]]

    plot_thresholds(y_test, y_pred, False, 'binary')
