from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np

import config

if __name__ == '__main__':
    data = config.get_seg_data('held_out')

    print("Loading the data")
    y = np.load(data['y'])
    y_true = np.load(data['y_true'])

    cluster_range = range(2, 6)
    y_pred = np.zeros((y_true.shape[0], 1))
    window_size = 20
    one_percent = int(y.shape[0] / 100) if y.shape[0] > 100 else 1

    for i in range(0, y.shape[0] - window_size, window_size):
        # for i in range(0, 40, window_size):
        y_select = y[i:(i + window_size), :]

        silhouette_avg_best = -1
        cluster_labels_best = None

        for n_clusters in cluster_range:
            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(y_select)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(y_select, cluster_labels)

            if silhouette_avg > silhouette_avg_best:
                silhouette_avg_best = silhouette_avg
                cluster_labels_best = cluster_labels

        for j in range(0, window_size - 1):
            if cluster_labels_best[j] != cluster_labels_best[j + 1]:
                y_pred[i + j] = 1

        if i != 0 and i % one_percent == 0:
            print("{0} % done".format(i / one_percent))

    P, R, F, S = prfs(y_true, y_pred, average='binary')

    print('F1 = %.3f (P = %.3f, R = %.3f)' % (F, P, R))
    # F1 = 0.419 (P = 0.301, R = 0.692)
