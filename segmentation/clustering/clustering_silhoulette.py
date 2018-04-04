from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np

import config
from segmentation.clustering.custom_clusterer import CustomClusterer
from utils import first_option, print_measurements

if __name__ == '__main__':
    custom_clusterer = first_option('Do you want to use custom implementation of clusterer [c] or k-means [k]?',
                                    'c', 'k')

    data = config.get_seg_data('test')

    print("Loading the data")

    # Predictions to undergo clustering
    X = np.load(data['y'])
    y_true = np.load(data['y_true_lm'])

    range_n_clusters = [2, 3, 4, 5, 6]
    y_pred = np.zeros((y_true.shape[0], 1))
    window_size = 20
    one_percent = int(X.shape[0] / 100) if X.shape[0] > 100 else 1

    print("Clustering...")
    for i in range(0, X.shape[0] - window_size, window_size):
        # for i in range(0, 40, window_size):
        X_select = X[i:(i + window_size), :]

        silhouette_avg_best = -1
        cluster_labels_best = None

        for n_clusters in range_n_clusters:
            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            if custom_clusterer:
                clusterer = CustomClusterer(n_clusters=n_clusters)
            else:
                clusterer = KMeans(n_clusters=n_clusters, random_state=10)

            cluster_labels = clusterer.fit_predict(X_select)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X_select, cluster_labels)

            if silhouette_avg > silhouette_avg_best:
                silhouette_avg_best = silhouette_avg
                cluster_labels_best = cluster_labels

        for j in range(0, window_size - 1):
            if cluster_labels_best[j] != cluster_labels_best[j + 1]:
                y_pred[i + j] = 1

        if i != 0 and i % one_percent == 0:
            print("{0} % done".format(i / one_percent))

    print_measurements(y_true, y_pred)