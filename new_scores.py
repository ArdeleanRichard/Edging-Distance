import numpy as np
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import time

from datasets import create_data4
from distance import max_edge_in_guided_path, centre_from_data
from labelling import diagonal_line, assign_labels_by_given_line, vertical_line, horizontal_line




def new_silhouette_score(X, labels, k=5, lookahead=3, debug=False):
    """
    Calculate the Silhouette score for a given set of labels.

    Parameters:
    - X: ndarray, shape (n_samples, n_features)
        The input data.
    - labels: ndarray, shape (n_samples,)
        Cluster labels for each data point.

    Returns:
    - db_score: float
        The Davies-Bouldin score.
    """
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1:
        return 0

    cluster_means = np.array([centre_from_data(X[labels == i]) for i in unique_labels])

    distance_matrix = np.zeros((n_samples, n_clusters))
    for sid, sample in enumerate(X):
        for cid, cc in enumerate(cluster_means):
            dist = max_edge_in_guided_path(X, sample, cc, k, lookahead, debug)
            distance_matrix[sid, cid] = dist

    # Compute the intra-cluster distances
    intra_cluster_distances = np.zeros(n_samples)
    for i in range(n_clusters):
        intra_cluster_distances[labels == unique_labels[i]] = np.mean(distance_matrix[labels == unique_labels[i], i])

    # Compute the inter-cluster distances for each sample
    inter_cluster_distances = np.min(distance_matrix + np.where(labels[:, np.newaxis] == unique_labels, np.inf, 0), axis=1)

    # Compute the silhouette coefficient for each sample
    silhouette_coefficients = (inter_cluster_distances - intra_cluster_distances) / np.maximum(inter_cluster_distances, intra_cluster_distances)

    # Compute the mean silhouette score over all samples
    mean_silhouette_score = np.mean(silhouette_coefficients)

    return mean_silhouette_score


def new_davies_bouldin_score(X, labels, k=5, lookahead=3, debug=False):
    """
    Calculate the Davies-Bouldin score for a given set of labels.

    Parameters:
    - X: ndarray, shape (n_samples, n_features)
        The input data.
    - labels: ndarray, shape (n_samples,)
        Cluster labels for each data point.

    Returns:
    - db_score: float
        The Davies-Bouldin score.
    """
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Calculate cluster means
    cluster_means = np.array([centre_from_data(X[labels == i]) for i in unique_labels])

    # Calculate cluster distances
    cluster_distances = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                # cluster_distances[i, j] = np.linalg.norm(cluster_means[i] - cluster_means[j])
                cluster_distances[i, j] = max_edge_in_guided_path(X, cluster_means[i], cluster_means[j], k, lookahead, debug)

    # Calculate cluster-wise scatter
    cluster_scatter = np.array([
        np.mean([
            max_edge_in_guided_path(X, sample, cluster_means[i], k, lookahead, debug)
            for sample in X[labels==i]
        ]) for i in range(n_clusters)
    ])

    # Calculate Davies-Bouldin index
    db_index = 0.0
    for i in range(n_clusters):
        max_similarity = -np.inf
        for j in range(n_clusters):
            if i != j:
                similarity = (cluster_scatter[i] + cluster_scatter[j]) / cluster_distances[i, j]
                if similarity > max_similarity:
                    max_similarity = similarity
        db_index += max_similarity
    db_index /= n_clusters

    return db_index


def new_calinski_harabasz_score(X, labels, k=5, lookahead=3, debug=False):
    """
    Calculate the Calinski-Harabasz score for a given set of labels.

    Parameters:
    - X: ndarray, shape (n_samples, n_features)
        The input data.
    - labels: ndarray, shape (n_samples,)
        Cluster labels for each data point.

    Returns:
    - ch_index: float
        The Calinski-Harabasz score.
    """
    n_samples, n_features = X.shape
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Calculate cluster means
    cluster_means = np.array([centre_from_data(X[labels == i]) for i in unique_labels])

    # Calculate overall mean
    overall_mean = centre_from_data(X)

    # Calculate between-cluster sum of squares
    # between_cluster_ss = np.sum([np.sum((cluster_means[i] - overall_mean) ** 2) * np.sum(labels == i) for i in range(n_clusters)])
    between_cluster_ss = np.sum([max_edge_in_guided_path(X, cluster_means[i], overall_mean, k, lookahead, debug) * np.sum(labels == i) for i in range(n_clusters)])


    # Calculate within-cluster sum of squares
    # within_cluster_ss = np.sum([np.sum((X[labels == i] - cluster_means[i]) ** 2) for i in range(n_clusters)])
    within_cluster_ss = np.sum([
            max_edge_in_guided_path(X, sample, cluster_means[i], k, lookahead, debug)
            for i in range(n_clusters)
        for sample in X[labels == i]
    ])

    # Calculate Calinski-Harabasz index
    ch_index = (between_cluster_ss / (n_clusters - 1)) / (within_cluster_ss / (n_samples - n_clusters))

    return ch_index



if __name__ == '__main__':
    n_samples = 1000
    X, y = create_data4(n_samples)


    X = MinMaxScaler((-1, 1)).fit_transform(X)
    X, y = shuffle(X, y, random_state=7)

    line = diagonal_line(X)
    dp = assign_labels_by_given_line(X, line)

    line = vertical_line(0)
    vl = assign_labels_by_given_line(X, line)

    line = horizontal_line(0)
    hl = assign_labels_by_given_line(X, line)

    k = 5
    la = 20
    y = np.copy(vl)


    start = time.time()
    score = silhouette_score(X, y)
    print(f"SS: {score} in {time.time() - start:.2f}s")
    start = time.time()
    score = new_silhouette_score(X, y, debug=True)
    print(f"NSS: {score} in {time.time() - start:.2f}s")
    print()


    start = time.time()
    score = davies_bouldin_score(X, y)
    print(f"DBS: {score} in {time.time() - start:.2f}s")
    start = time.time()
    score = new_davies_bouldin_score(X, y, k=k, lookahead=la, debug=True)
    print(f"NDBS: {score} in {time.time() - start:.2f}s")
    print()

    start = time.time()
    score = calinski_harabasz_score(X, y)
    print(f"CHS: {score} in {time.time() - start:.2f}s")
    start = time.time()
    score = new_calinski_harabasz_score(X, y, debug=True)
    print(f"NCHS: {score} in {time.time() - start:.2f}s")
    print()


