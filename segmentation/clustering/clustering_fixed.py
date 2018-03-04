from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support as prfs
import numpy as np

import config
from segmentation.clustering.custom_clusterer import CustomClusterer
from utils import first_option

if __name__ == '__main__':
    custom_clusterer = first_option('Do you want to custom implementation of clusterer [c] or k-means [k]?', 'c', 'k')
    n_clusters = int(input("Enter number of clusters: "))

    data = config.get_seg_data('test')

    print("Loading the data")
    X = np.load(data['y'])
    y_true = np.load(data['y_true_lm'])

    cluster_range = range(2, 6)
    y_pred = np.zeros((y_true.shape[0], 1))
    window_size = 20
    one_percent = int(X.shape[0] / 100) if X.shape[0] > 100 else 1

    print("Clustering...")
    for i in range(0, X.shape[0] - window_size, window_size):
        # for i in range(0, 40, window_size):
        X_select = X[i:(i + window_size), :]

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        if custom_clusterer:
            clusterer = CustomClusterer(n_clusters=n_clusters)
        else:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)

        cluster_labels = clusterer.fit_predict(X_select)

        for j in range(0, window_size - 1):
            if cluster_labels[j] != cluster_labels[j + 1]:
                y_pred[i + j] = 1

        if i != 0 and i % one_percent == 0:
            print("{0} % done".format(i / one_percent))

    P, R, F, S = prfs(y_true, y_pred, average='binary')

    print('n_clusters: %.0f, F1 = %.3f (P = %.3f, R = %.3f)' % (n_clusters, F, P, R))
