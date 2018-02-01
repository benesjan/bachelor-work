import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import MinMaxScaler

import config


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


def slide_window(y_n, window_size=4):
    y_p = np.zeros((y_n.shape[0], 1))
    for i in range(window_size, y_n.shape[0]):
        y_p[i] = np.abs(np.mean(y_n[i - window_size:i]) - y_n[i])
    return y_p


def neighbourhood_difference(y_n, epsilon=2):
    y_p = np.zeros((y_n.shape[0], 1))
    for i in range(epsilon, y_n.shape[0] - epsilon):
        avg = (np.sum(y_n[(i - epsilon): i]) + np.sum(y_n[i + 1: (i + 1 + epsilon)])) / (2 * epsilon)
        y_p[i] = np.abs(avg - y_n[i])
    return y_p


# Optimal threshold discovery
# if __name__ == '__main__':
#     data = config.get_seg_data('held_out')
#
#     print("Loading the data")
#     y = np.load(data['y'])
#     y_true = np.load(data['y_true_lm'])
#
#     # y_norms = compute_euclidean_distance(y)
#     y_norms = compute_cosine_distance(y)
#
#     # best result euclidean: threshold = 0.5, F1 = 0.546 (P = 0.507, R = 0.592)
#     # best result cosine: threshold = 0.92, F1 = 0.708 (P = 0.709, R = 0.706)
#     y_pred = y_norms
#
#     # best result: window = 4, threshold = 0.42, F1 = 0.419 (P = 0.339, R = 0.550)
#     # y_pred = slide_window(y_norms)
#
#     # best result: epsilon = 2, threshold = 0.46, F1 = 0.515 (P = 0.468, R = 0.573)
#     # y_pred = neighbourhood_difference(y_norms)
#
#     plot_thresholds(y_true, y_pred, False, 'binary')

# Measurements on test data
if __name__ == '__main__':
    data = config.get_seg_data('test')

    print("Loading the data")
    y = np.load(data['y'])
    y_true = np.load(data['y_true_lm'])

    T = 0.5
    y_pred = compute_euclidean_distance(y) > T
    P, R, F, S = prfs(y_true, y_pred, average='binary')
    print('compute_euclidean_distance: threshold = %.2f, F1 = %.3f (P = %.3f, R = %.3f)' % (T, F, P, R))

    y_norms = compute_cosine_distance(y)

    T = 0.92
    y_pred = y_norms > T
    P, R, F, S = prfs(y_true, y_pred, average='binary')
    print('compute_cosine_distance: threshold = %.2f, F1 = %.3f (P = %.3f, R = %.3f)' % (T, F, P, R))

    T = 0.42
    window_size = 4
    y_pred = slide_window(y_norms, window_size=window_size) > T
    P, R, F, S = prfs(y_true, y_pred, average='binary')
    print('slide_window: threshold = %.2f, window_size = %.0f, F1 = %.3f (P = %.3f, R = %.3f)' % (
    T, window_size, F, P, R))

    epsilon = 2
    T = 0.46
    y_pred = neighbourhood_difference(y_norms, epsilon=epsilon) > T
    P, R, F, S = prfs(y_true, y_pred, average='binary')
    print('neighbourhood_difference: threshold = %.2f, epsilon = %.0f, F1 = %.3f (P = %.3f, R = %.3f)' % (T, epsilon, F, P, R))
