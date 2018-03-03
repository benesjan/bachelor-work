import numpy as np
from sklearn.svm import LinearSVC, SVC

import config
from segmentation.distance_based_methods import compute_distance
from utils import save_pickle, first_option, load_sparse_csr

if __name__ == '__main__':
    data = config.get_seg_data('train')

    if first_option('Do you want to use linear [l] or RBF [r] kernel?', 'l', 'r'):
        path = config.classifier_linear
        classifier = LinearSVC(random_state=0)
    else:
        path = config.classifier_rbf
        classifier = SVC(random_state=0, kernel='rbf')

    y_true = np.load(data['y_true_lm'])

    print("Loading x")
    x = load_sparse_csr(data['x'])

    print("Computing cosine distance")
    x_dists = compute_distance(x)

    print("Classifier training")
    classifier.fit(x_dists, y_true)

    print("Saving th classifier to: " + path)
    save_pickle(path, classifier)
