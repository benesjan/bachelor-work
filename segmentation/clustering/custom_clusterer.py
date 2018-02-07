import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from segmentation.distance_based_methods import compute_distance


class CustomClusterer:
    n_clusters = None

    def __init__(self, n_clusters=4):
        self._n_clusters = n_clusters

    def fit_predict(self, y):
        # Append value at the position of last vector in the y selection
        dists = np.append(compute_distance(y, cosine_distances, False), 1000)

        n_clusters_current = y.shape[0]

        while n_clusters_current != self.n_clusters:
            cluster1_start = np.argmin(dists)

            cluster1_size = self._get_cluster_size(dists, cluster1_start)

            cluster2_start = cluster1_start + cluster1_size

            cluster2_size = self._get_cluster_size(dists, cluster2_start)

            new_cluster_size = cluster1_size + cluster2_size

            # Compute new cluster center
            y[cluster1_start, :] = np.sum([y[cluster1_start, :] * cluster1_size, y[cluster2_start, :] * cluster2_size],
                                          axis=0) / new_cluster_size

            # Set the invalidated distances to 10
            dists[cluster2_start] = 10

            # Set the invalidated predictions to 0
            # just for debugging, because clusters are mapped using the vector of distances
            # y[cluster2_start, :] = 0

            next_cluster_center = cluster1_start + new_cluster_size
            if next_cluster_center == dists.shape[0]:
                dists[cluster1_start] = 100
            else:
                # Compute new distance between the new cluster center and the next cluster center
                dists[cluster1_start] = cosine_distances(y[cluster1_start:cluster1_start + 1, :],
                                                         y[new_cluster_size:new_cluster_size + 1, :])

            n_clusters_current -= 1

        # Convert cluster distances to cluster labels
        label = 0
        labels = np.zeros((dists.shape[0], 1))
        for j in range(0, dists.shape[0]):
            if dists[j] != 10:
                label += 1
            labels[j] = label

        return labels

    def _get_cluster_size(self, dists, cluster_start):
        # find the index of next valid distance
        assert dists[cluster_start] != 10, "invalid cluster start"

        cluster_size = 1
        for j in range(cluster_start + 1, dists.shape[0]):
            if dists[j] != 10:
                break
            cluster_size += 1

        return cluster_size
