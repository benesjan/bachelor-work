import numpy as np
from matplotlib import pyplot, rc
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.preprocessing import MinMaxScaler

from custom_imports import config
from custom_imports.utils import load_sparse_csr


def compute_norm(y):
    """
    :param y: prediction matrix
    :return: vector with euclidean norms of every 2 consecutive vectors within the prediction matrix
    """
    y_n = np.zeros((y.shape[0] - 1, 1))
    for i in range(y.shape[0] - 1):
        y_n[i] = np.linalg.norm(y[i, :] - y[i + 1, :])
    return MinMaxScaler().fit_transform(y_n)


def plot_thresholds(y_true, y_raw):
    threshold_array = np.arange(0, 1.0, 0.01)

    values = np.ones((threshold_array.shape[0], 4), dtype=np.float)

    optimal_position = [-100, 0, 0, 0]

    for i, T in enumerate(threshold_array):
        y_pred = y_raw > T

        P, R, F, S = prfs(y_true, y_pred, average='binary')
        values[i, :] = [T, F, P, R]

        print('threshold = %.2f, F1 = %.3f (P = %.3f, R = %.3f)' % (T, F, P, R))

        if F > optimal_position[1]:
            optimal_position = values[i, :]

    rc('font', family='Arial')
    pyplot.plot(values[:, 0], values[:, 1:4])
    pyplot.legend(['F-measure', 'Precision', 'Recall'])

    pyplot.grid()
    pyplot.scatter(optimal_position[0], optimal_position[1], marker="x", s=300, linewidth=3.3)
    pyplot.annotate('[%.2f, %.2f]' % (optimal_position[0], optimal_position[1]),
                    [optimal_position[0], optimal_position[1] - 0.055])

    pyplot.xlim([0, 1])
    pyplot.ylim([0, 1])

    pyplot.title('Vývoj přesnosti, úplnosti a F-míry v závislosti na prahu')
    pyplot.xlabel('Práh')

    pyplot.show()


if __name__ == '__main__':
    data = config.get_seg_data('held_out')

    print("Loading the data")
    x = load_sparse_csr(data['x'])
    y = np.load(data['y'])
    y_true = np.load(data['y_true'])

    y_norms = compute_norm(y)

    plot_thresholds(y_true, y_norms)
