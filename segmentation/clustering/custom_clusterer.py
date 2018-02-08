import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from segmentation.distance_based_methods import compute_distance


class CustomClusterer:
    n_clusters = None

    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters

    def fit_predict(self, y):
        # Append value at the position of last vector in the y selection
        dists = np.append(compute_distance(y, cosine_distances, False), 1000)

        n_clusters_current = y.shape[0]

        while n_clusters_current != self.n_clusters:
            cluster1_center = np.argmin(dists)

            cluster1_size = self._get_cluster_size(dists, cluster1_center)

            cluster2_center = cluster1_center + cluster1_size

            cluster2_size = self._get_cluster_size(dists, cluster2_center)

            new_cluster_size = cluster1_size + cluster2_size

            # Compute new cluster center
            y[cluster1_center, :] = np.sum(
                [y[cluster1_center, :] * cluster1_size, y[cluster2_center, :] * cluster2_size],
                axis=0) / new_cluster_size

            # Set the invalidated distances to 10
            dists[cluster2_center] = 10

            # Set the invalidated predictions to 0
            # just for debugging, because clusters are mapped using the vector of distances
            # y[cluster2_center, :] = 0

            # Recompute the distance to the previous cluster if the current cluster is not at the beginning of sequence
            if cluster1_center != 0:
                prev_cluster_center = self._get_previous_cluster_center(dists, cluster1_center)
                dists[prev_cluster_center] = cosine_distances(y[prev_cluster_center:prev_cluster_center + 1, :],
                                                              y[cluster1_center:cluster1_center + 1, :])

            # Recompute the distance to the next cluster
            # If the current cluster is at the end of sequence set big dummy value
            next_cluster_center = cluster1_center + new_cluster_size
            if next_cluster_center == dists.shape[0]:
                dists[cluster1_center] = 100
            else:
                dists[cluster1_center] = cosine_distances(y[cluster1_center:cluster1_center + 1, :],
                                                          y[next_cluster_center:next_cluster_center + 1, :])

            n_clusters_current -= 1

        # Convert cluster distances to cluster labels
        label = 0
        labels = np.zeros((dists.shape[0], 1))
        for j in range(0, dists.shape[0]):
            if dists[j] != 10:
                label += 1
            labels[j] = label

        return labels

    def _get_cluster_size(self, dists, center):
        assert dists[center] != 10, "Invalid cluster center"

        cluster_size = 1
        for j in range(center + 1, dists.shape[0]):
            if dists[j] != 10:
                break
            cluster_size += 1

        return cluster_size

    def _get_previous_cluster_center(self, dists, center):
        assert center != 0, "Invalid cluster center"

        # Iterate from (current center -1) to 0
        for j in range(center - 1, -1, -1):
            if dists[j] != 10:
                return j
