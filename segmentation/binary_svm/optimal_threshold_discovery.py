import numpy as np

import config
from segmentation.distance_based_methods import compute_distance
from utils import load_pickle, first_option, plot_thresholds, load_sparse_csr

if __name__ == '__main__':
    data = config.get_seg_data('held_out')

    if first_option('Do you want to use linear [l] or RBF [r] kernel?', 'l', 'r'):
        classifier = load_pickle(config.classifier_linear)
        interval = (-1, 1)
    else:
        classifier = load_pickle(config.classifier_rbf)
        interval = (-10, 0)

    y_true = np.load(data['y_true_lm'])

    print("Loading x")
    x = load_sparse_csr(data['x'])

    print("Computing cosine distance")
    x_norms = compute_distance(x)

    print("Predicting")
    y_pred = classifier.decision_function(x_norms)

    plot_thresholds(y_true, y_pred, False, 'binary', interval)
