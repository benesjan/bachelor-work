import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler

import config
from classification.optimal_threshold_discovery import plot_thresholds
from utils import load_sparse_csr


def compute_euclidean_distance(y):
    """
    :param y: prediction matrix
    :return: vector with euclidean distances of every 2 consecutive vectors within the prediction matrix
    """
    y_n = np.zeros((y.shape[0] - 1, 1))
    for i in range(y.shape[0] - 1):
        y_n[i] = np.linalg.norm(y[i, :] - y[i + 1, :])
    return MinMaxScaler().fit_transform(y_n)


def compute_cosine_distance(y):
    """
    Much better performance then euclidean distance - Euclidean distance is susceptible to documents being clustered
    by their L2-norm (magnitude, in the 2 dimensional case) instead of direction. I.e. vectors with quite different
    directions would be clustered because their distances from origin are similar.
    :param y: prediction matrix
    :return: vector with cosine distances of every 2 consecutive vectors within the prediction matrix
    """
    y_n = np.zeros((y.shape[0] - 1, 1))
    for i in range(y.shape[0] - 1):
        y_n[i] = cosine_distances(y[i:(i + 1), :], y[(i + 1):(i + 2), :])

    # No need for scaling to interval <0, 1> because the vectors are only of positive values hence the cosine distance
    # is naturally bounded by <0, 1>
    return y_n


if __name__ == '__main__':
    data = config.get_seg_data('held_out')

    print("Loading the data")
    x = load_sparse_csr(data['x'])
    y = np.load(data['y'])
    y_true = np.load(data['y_true'])

    # y_norms = compute_euclidean_distance(y)
    y_norms = compute_cosine_distance(y)
    # plot_thresholds(y_true, y_norms, False, 'binary')

    window_size = 4

    y_pred = np.zeros((y_norms.shape[0], 1))
    for i in range(window_size, y_norms.shape[0]):
        y_pred[i] = np.abs(np.mean(y_norms[i - window_size:i]) - y_norms[i])

    plot_thresholds(y_true, y_pred, False, 'binary')
