import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs

import config
from segmentation.distance_based_methods import compute_distance
from utils import load_pickle, first_option, load_sparse_csr

if __name__ == '__main__':
    data = config.get_seg_data('test')

    if first_option('Do you want to use linear [l] or RBF [r] kernel?', 'l', 'r'):
        path = config.classifier_linear
        threshold = -0.55
    else:
        path = config.classifier_rbf
        threshold = -5

    classifier = load_pickle(path)

    y_true = np.load(data['y_true_lm'])

    print("Loading x")
    x = load_sparse_csr(data['x'])

    print("Computing cosine distance")
    x_norms = compute_distance(x)

    print("Predicting")
    y_pred = classifier.decision_function(x_norms) > threshold

    P, R, F, S = prfs(y_true, y_pred, average='binary')
    print('F1 = %.3f (P = %.3f, R = %.3f)' % (F, P, R))
