import numpy as np
from keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics.pairwise import cosine_distances

import config
from segmentation.distance_based_methods import compute_distance
from segmentation.lstm.lstm_utils import split_to_time_steps
from utils import first_option

if __name__ == '__main__':

    if first_option('Do you want to use the model trained on cosine distances [c] or on raw SVM predictions [r]?',
                    'c', 'r'):
        cosine = True
        model = load_model(config.lstm_model_1)
        T = 0.41
    else:
        cosine = False
        model = load_model(config.lstm_model_577)
        T = 0.48

    time_steps = model.get_config()[0]['config']['batch_input_shape'][1]

    test = config.get_seg_data('test')

    X_test = np.load(test['y'])
    y_test = np.load(test['y_true_lm'])

    if cosine:
        print("Computing the distances")
        X_test = compute_distance(X_test, cosine_distances)
    else:
        y_test = np.append(0, y_test)

    X = split_to_time_steps(X_test)
    y_true = split_to_time_steps(y_test)

    y_pred = model.predict(X) > T

    P, R, F, S = prfs(y_true.flatten(), y_pred.flatten(), average='binary')
    print('F1 = %.3f (P = %.3f, R = %.3f)' % (F, P, R))
