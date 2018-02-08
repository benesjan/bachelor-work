import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
from sklearn.preprocessing import MinMaxScaler

import config


def compute_distance(y, distance_func=cosine_distances, scale=True):
    """
    :param scale: Parameter which sets whether the distances should be scaled between 0 and 1
    :param distance_func: function which computes the distance between 2 vectors, has to follow metrics.pairwise API
    :param y: prediction matrix
    :return: vector with euclidean distances of every 2 consecutive vectors within the prediction matrix
    """
    y_n = np.zeros((y.shape[0] - 1, 1))
    for i in range(y.shape[0] - 1):
        y_n[i] = distance_func(y[i:(i + 1), :], y[(i + 1):(i + 2), :])

    return MinMaxScaler().fit_transform(y_n) if scale else y_n


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
#     print("Computing the distances")
#     # y_dists = compute_distance(y, euclidean_distances)
#     y_dists = compute_distance(y, cosine_distances)
#     # y_dists = compute_distance(y, manhattan_distances)
#
#     # best result euclidean: threshold = 0.5, F1 = 0.546 (P = 0.507, R = 0.592)
#     # best result cosine: threshold = 0.92, F1 = 0.708 (P = 0.709, R = 0.706)
#     # best result manhattan: threshold = 0.33, F1 = 0.537 (P = 0.474, R = 0.620)
#     y_pred = y_dists
#
#     # best result: window = 4, threshold = 0.42, F1 = 0.419 (P = 0.339, R = 0.550)
#     # Better result for very high windows, window = 1000, threshold = 0.43, F1 = 0.493 (P = 0.422, R = 0.591)
#     # y_pred = slide_window(y_dists)
#
#     # best result: epsilon = 2, threshold = 0.46, F1 = 0.515 (P = 0.468, R = 0.573)
#     # y_pred = neighbourhood_difference(y_dists)
#
#     plot_thresholds(y_true, y_pred, False, 'binary')

# Measurements on test data
if __name__ == '__main__':
    data = config.get_seg_data('test')

    print("Loading the data")
    y = np.load(data['y'])
    y_true = np.load(data['y_true_lm'])

    T = 0.5
    y_pred = compute_distance(y, euclidean_distances) > T
    P, R, F, S = prfs(y_true, y_pred, average='binary')
    print('euclidean distance: threshold = %.2f, F1 = %.3f (P = %.3f, R = %.3f)' % (T, F, P, R))

    y_dists = compute_distance(y, cosine_distances)

    T = 0.92
    y_pred = y_dists > T
    P, R, F, S = prfs(y_true, y_pred, average='binary')
    print('cosine distance: threshold = %.2f, F1 = %.3f (P = %.3f, R = %.3f)' % (T, F, P, R))

    T = 0.33
    y_pred = compute_distance(y, manhattan_distances) > T
    P, R, F, S = prfs(y_true, y_pred, average='binary')
    print('manhattan distance: threshold = %.2f, F1 = %.3f (P = %.3f, R = %.3f)' % (T, F, P, R))

    T = 0.42
    window_size = 4
    y_pred = slide_window(y_dists, window_size=window_size) > T
    P, R, F, S = prfs(y_true, y_pred, average='binary')
    print('slide_window, cosine distance: threshold = %.2f, window_size = %.0f, F1 = %.3f (P = %.3f, R = %.3f)'
          % (T, window_size, F, P, R))

    epsilon = 2
    T = 0.46
    y_pred = neighbourhood_difference(y_dists, epsilon=epsilon) > T
    P, R, F, S = prfs(y_true, y_pred, average='binary')
    print('neighbourhood_difference, cosine distance: threshold = %.2f, epsilon = %.0f, F1 = %.3f (P = %.3f, R = %.3f)'
          % (T, epsilon, F, P, R))
