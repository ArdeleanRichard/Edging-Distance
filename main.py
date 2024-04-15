import time

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from datasets import create_data3, create_data6, create_data7, create_data4, create_data2, create_data1, create_data5, create_data8, create_data4_3d, create_data5_3d, create_data3_3d, create_iris, create_set1, create_set3d, create_set2
from distance import max_edge_in_guided_path_for_kmeans
from label_map import LABEL_COLOR_MAP

from labelling import diagonal_line, vertical_line, assign_labels_by_given_line, horizontal_line, assign_labels_by_given_plane
from new_kmeans import NewKMeansClustering
from new_scores import new_silhouette_score, new_davies_bouldin_score, new_calinski_harabasz_score
from np_kmeans import KMeansClustering


def run_score_set1(datasets, metric, k, la, plot=False):
    for i_dataset, (X, gt) in enumerate(datasets):
        X, gt = shuffle(X, gt, random_state=7)
        X = MinMaxScaler((-1, 1)).fit_transform(X)

        line = diagonal_line(X)
        dp = assign_labels_by_given_line(X, line)

        line = vertical_line(0)
        vl = assign_labels_by_given_line(X, line)

        line = horizontal_line(0)
        hl = assign_labels_by_given_line(X, line)

        rl = np.random.randint(0, len(np.unique(gt)), size=len(X))

        if metric == 'SS':
            print(f"{silhouette_score(X, gt):.3f}, {new_silhouette_score(X, gt, k=k, lookahead=la):.3f}, \t\t"
                  f"{silhouette_score(X, dp):.3f}, {new_silhouette_score(X, dp, k=k, lookahead=la):.3f}, \t\t"
                  f"{silhouette_score(X, vl):.3f}, {new_silhouette_score(X, vl, k=k, lookahead=la):.3f}, \t\t"
                  f"{silhouette_score(X, hl):.3f}, {new_silhouette_score(X, hl, k=k, lookahead=la):.3f}, \t\t"
                  f"{silhouette_score(X, rl):.3f}, {new_silhouette_score(X, rl, k=k, lookahead=la):.3f}, \t\t")
        if metric == 'DBS':
            print(f"{davies_bouldin_score(X, gt):.3f}, {new_davies_bouldin_score(X, gt, k=k, lookahead=la, debug=False):.3f}, \t\t"
                  f"{davies_bouldin_score(X, dp):.3f}, {new_davies_bouldin_score(X, dp, k=k, lookahead=la, debug=False):.3f}, \t\t"
                  f"{davies_bouldin_score(X, vl):.3f}, {new_davies_bouldin_score(X, vl, k=k, lookahead=la, debug=False):.3f}, \t\t"
                  f"{davies_bouldin_score(X, hl):.3f}, {new_davies_bouldin_score(X, hl, k=k, lookahead=la, debug=False):.3f}, \t\t"
                  f"{davies_bouldin_score(X, rl):.3f}, {new_davies_bouldin_score(X, rl, k=k, lookahead=la, debug=False):.3f}, \t\t")
        if metric == 'CHS':
            print(f"{calinski_harabasz_score(X, gt):.2f}, {new_calinski_harabasz_score(X, gt, k=k, lookahead=la):.2f}, \t\t"
                  f"{calinski_harabasz_score(X, dp):.2f}, {new_calinski_harabasz_score(X, dp, k=k, lookahead=la):.2f}, \t\t"
                  f"{calinski_harabasz_score(X, vl):.2f}, {new_calinski_harabasz_score(X, vl, k=k, lookahead=la):.2f}, \t\t"
                  f"{calinski_harabasz_score(X, hl):.2f}, {new_calinski_harabasz_score(X, hl, k=k, lookahead=la):.2f}, \t\t"
                  f"{calinski_harabasz_score(X, rl):.2f}, {new_calinski_harabasz_score(X, rl, k=k, lookahead=la):.2f}, \t\t")

        if plot:
            for name, labels in zip(["gt", "dp", "vl", "hl", "rl"], [gt, dp, vl, hl, rl]):
                label_color = [LABEL_COLOR_MAP[i] for i in labels]
                plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
                plt.savefig(f"./figs/data/data{i_dataset + 1}_{name}.svg")
                plt.savefig(f"./figs/data/data{i_dataset + 1}_{name}.png")
                plt.close()

    print()
    # plt.show()


def run_scores_set1(plot=False):
    n_samples = 1000
    datasets = create_set1(n_samples)

    run_scores_set1(datasets, "", k=0, la=0, plot=plot)

    metric = "SS"
    run_scores_set1(datasets, metric, k=3, la=10)
    # run_scores_set1(datasets, metric, n_neighbours=15, la=10)
    # run_scores_set1(datasets, metric, n_neighbours=5, la=3)

    metric = "DBS"
    run_scores_set1(datasets, metric, k=3, la=10)
    # run_scores_set1(datasets, metric, n_neighbours=15, la=10)
    # run_scores_set1(datasets, metric, n_neighbours=5, la=3)

    metric = "CHS"
    run_scores_set1(datasets, metric, k=3, la=10)
    # run_scores_set1(datasets, metric, n_neighbours=15, la=10)
    # run_scores_set1(datasets, metric, n_neighbours=5, la=3)



def run_score_set2(datasets, metric, k, la, plot=False):
    for i_dataset, (X, gt) in enumerate(datasets):
        X, gt = shuffle(X, gt, random_state=7)
        X = MinMaxScaler((-1, 1)).fit_transform(X)

        rl = np.random.randint(0, len(np.unique(gt)), size=len(X))

        if metric == 'SS':
            print(f"{silhouette_score(X, gt):.3f}, {new_silhouette_score(X, gt, k=k, lookahead=la):.3f}, \t\t"
                  f"{silhouette_score(X, rl):.3f}, {new_silhouette_score(X, rl, k=k, lookahead=la):.3f}, \t\t")
        if metric == 'DBS':
            print(f"{davies_bouldin_score(X, gt):.3f}, {new_davies_bouldin_score(X, gt, k=k, lookahead=la, debug=False):.3f}, \t\t"
                  f"{davies_bouldin_score(X, rl):.3f}, {new_davies_bouldin_score(X, rl, k=k, lookahead=la, debug=False):.3f}, \t\t")
        if metric == 'CHS':
            print(f"{calinski_harabasz_score(X, gt):.2f}, {new_calinski_harabasz_score(X, gt, k=k, lookahead=la):.2f}, \t\t"
                  f"{calinski_harabasz_score(X, rl):.2f}, {new_calinski_harabasz_score(X, rl, k=k, lookahead=la):.2f}, \t\t")

        if plot:
            for name, labels in zip(["gt", "rl"], [gt, rl]):
                label_color = [LABEL_COLOR_MAP[i] for i in labels]
                plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
                plt.savefig(f"./figs/data/data{i_dataset + 1}_{name}.svg")
                plt.savefig(f"./figs/data/data{i_dataset + 1}_{name}.png")
                plt.close()

    print()
    # plt.show()

def run_scores_set2(plot=False):
    datasets = create_set2()

    run_score_set2(datasets, "", k=0, la=0, plot=plot)

    metric = "SS"
    run_score_set2(datasets, metric, k=1, la=50)
    # run_score(datasets, metric, n_neighbours=15, la=10)
    # run_score(datasets, metric, n_neighbours=5, la=3)

    metric = "DBS"
    run_score_set2(datasets, metric, k=3, la=20)
    # run_score(datasets, metric, n_neighbours=15, la=10)
    # run_score(datasets, metric, n_neighbours=5, la=3)

    metric = "CHS"
    run_score_set2(datasets, metric, k=3, la=20)
    # run_score(datasets, metric, n_neighbours=15, la=10)
    # run_score(datasets, metric, n_neighbours=5, la=3)





def run_score_3d(metric, k, la, plot=False):
    n_samples = 1000

    datasets = create_set3d(n_samples)

    for i_dataset, (X, gt) in enumerate(datasets):
        X = MinMaxScaler((-1, 1)).fit_transform(X)
        rl = np.random.randint(0, len(np.unique(gt)), size=len(X))

        dp = assign_labels_by_given_plane(X, plane=((-1, -1, 1), (1,1,1), (-1,-1,-1)))
        vp = assign_labels_by_given_plane(X, plane="vertical")
        hp = assign_labels_by_given_plane(X, plane="horizontal")

        if metric == 'SS':
            print(f"{silhouette_score(X, gt):.3f}, {new_silhouette_score(X, gt, k=k, lookahead=la):.3f}, \t\t"
                  f"{silhouette_score(X, dp):.3f}, {new_silhouette_score(X, dp, k=k, lookahead=la):.3f}, \t\t"
                  f"{silhouette_score(X, vp):.3f}, {new_silhouette_score(X, vp, k=k, lookahead=la):.3f}, \t\t"
                  f"{silhouette_score(X, hp):.3f}, {new_silhouette_score(X, hp, k=k, lookahead=la):.3f}, \t\t"
                  f"{silhouette_score(X, rl):.3f}, {new_silhouette_score(X, rl, k=k, lookahead=la):.3f}, \t\t")
        if metric == 'DBS':
            print(f"{davies_bouldin_score(X, gt):.3f}, {new_davies_bouldin_score(X, gt, k=k, lookahead=la, debug=False):.3f}, \t\t"
                  f"{davies_bouldin_score(X, dp):.3f}, {new_davies_bouldin_score(X, dp, k=k, lookahead=la, debug=False):.3f}, \t\t"
                  f"{davies_bouldin_score(X, vp):.3f}, {new_davies_bouldin_score(X, vp, k=k, lookahead=la, debug=False):.3f}, \t\t"
                  f"{davies_bouldin_score(X, hp):.3f}, {new_davies_bouldin_score(X, hp, k=k, lookahead=la, debug=False):.3f}, \t\t"
                  f"{davies_bouldin_score(X, rl):.3f}, {new_davies_bouldin_score(X, rl, k=k, lookahead=la, debug=False):.3f}, \t\t")
        if metric == 'CHS':
            print(f"{calinski_harabasz_score(X, gt):.2f}, {new_calinski_harabasz_score(X, gt, k=k, lookahead=la):.2f}, \t\t"
                  f"{calinski_harabasz_score(X, dp):.2f}, {new_calinski_harabasz_score(X, dp, k=k, lookahead=la):.2f}, \t\t"
                  f"{calinski_harabasz_score(X, vp):.2f}, {new_calinski_harabasz_score(X, vp, k=k, lookahead=la):.2f}, \t\t"
                  f"{calinski_harabasz_score(X, hp):.2f}, {new_calinski_harabasz_score(X, hp, k=k, lookahead=la):.2f}, \t\t"
                  f"{calinski_harabasz_score(X, rl):.2f}, {new_calinski_harabasz_score(X, rl, k=k, lookahead=la):.2f}, \t\t")

        if plot:
            for name, labels in zip(["gt", "dp", "vp", "hp", "rl"], [gt, dp, vp, hp, rl]):
                label_color = [LABEL_COLOR_MAP[i] for i in labels]
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.view_init(elev=14., azim=-75)
                ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', c=label_color, edgecolors='k', s=25, alpha=0.5)
                # plt.savefig(f"./figs/data/data{i_dataset + 3}_3d_{name}.svg")
                plt.savefig(f"./figs/data3d/data{i_dataset + 3}_3d_{name}.png")
                plt.close()


def run_scores_set3d(plot=False):
    run_score_3d("", k=0, la=0, plot=plot)

    metric = "SS"
    run_score_3d(metric, k=3, la=10)
    run_score_3d(metric, k=15, la=10)
    run_score_3d(metric, k=5, la=3)

    metric = "DBS"
    run_score_3d(metric, k=3, la=50)
    run_score_3d(metric, k=15, la=10)
    run_score_3d(metric, k=5, la=3)

    metric = "CHS"
    run_score_3d(metric, k=3, la=50)
    run_score_3d(metric, k=15, la=10)
    run_score_3d(metric, k=5, la=3)


def analyze_time_scores_examples():
    # for n_samples in [100, 500, 1000, 5000, 10000]:
    #     X, gt = create_data3(n_samples)
    #     ss_start = time.time()
    #     ss_score = silhouette_score(X, gt)
    #     ss_time = time.time() - ss_start
    #
    #     nss_start = time.time()
    #     nss_score = new_silhouette_score(X, gt, k=5, lookahead=10, debug=False)
    #     nss_time = time.time() - nss_start
    #     print(f"D3 with {n_samples}samples, SS: {ss_score:.3f} in {ss_time:.3f}s, NSS: {nss_score:.3f} in {nss_time:.3f}s")
    #     print()

    for n_samples in [5000, 10000]:
        X, gt = create_data3(n_samples)
        ss_start = time.time()
        ss_score = silhouette_score(X, gt)
        ss_time = time.time() - ss_start

        nss_start = time.time()
        nss_score = new_silhouette_score(X, gt, k=50, lookahead=10, debug=False)
        nss_time = time.time() - nss_start
        print(f"D3 with {n_samples}samples, SS: {ss_score:.3f} in {ss_time:.3f}s, NSS: {nss_score:.3f} in {nss_time:.3f}s")
        print()



def analyze_time_scores_features():
    n_samples = 1000
    for n_features in [2,3,4,5,6]:
        X, gt = create_data3(n_samples, n_features)
        ss_start = time.time()
        ss_score = silhouette_score(X, gt)
        ss_time = time.time() - ss_start


        nss_start = time.time()
        nss_score = new_silhouette_score(X, gt, debug=True)
        nss_time = time.time() - nss_start
        print(f"D3 with {n_features}features, SS: {ss_score:.3f} in {ss_time:.3f}s, NSS: {nss_score:.3f} in {nss_time:.3f}s")
        print()



def run_kmeans_set1():
    n_samples = 1000
    datasets = create_set1(n_samples)

    for i_dataset, (X, gt) in enumerate(datasets):
        label_color = [LABEL_COLOR_MAP[i] for i in gt]
        plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
        # plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_gt.svg")
        plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_gt.png")
        plt.close()

        Kmeans = KMeansClustering(X, len(np.unique(gt)))
        k_labels = Kmeans.fit(X)

        km = KMeans(n_clusters=2, ).fit(X)
        k_labels = km.labels_

        label_color = [LABEL_COLOR_MAP[i] for i in k_labels]
        # plt.title(f"K-Means on D{i_dataset+1}")
        plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
        # plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_k.svg")
        plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_k.png")
        # plt.show()
        plt.close()




        newKmeans = NewKMeansClustering(X, len(np.unique(gt)), neighbors=5, lookahead=20)
        nk_labels, centroids = newKmeans.fit(X)

        label_color = [LABEL_COLOR_MAP[i] for i in nk_labels]
        # plt.title(f"NewK-Means on D{i_dataset+1}")
        plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
        # plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_nk.svg")
        plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_nk.png")
        # plt.show()
        plt.close()



        sc = SpectralClustering(n_clusters=len(np.unique(gt)), eigen_solver="arpack", affinity="nearest_neighbors", random_state=0).fit(X)
        s_labels = sc.labels_

        label_color = [LABEL_COLOR_MAP[i] for i in s_labels]
        # plt.title(f"NewK-Means on D{i_dataset+1}")
        plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
        # plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_nk.svg")
        plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_sc.png")
        # plt.show()
        plt.close()

        print(f"{adjusted_rand_score(k_labels, gt):.3f}, {adjusted_rand_score(nk_labels, gt):.3f}, {adjusted_rand_score(s_labels, gt):.3f}")
        print(f"{adjusted_mutual_info_score(k_labels, gt):.3f}, {adjusted_mutual_info_score(nk_labels, gt):.3f}, {adjusted_mutual_info_score(s_labels, gt):.3f}")
        print()

        # for id, x in enumerate(X):
        #     distances = np.sqrt(np.sum((X - x) ** 2, axis=1))
        #     sorted_distances = np.argsort(distances)
        #     nearest_indices = sorted_distances[:5]
        #
        #     if np.any(nk_labels[nearest_indices] != nk_labels[id]):
        #
        #         dist_to_centres = []
        #         path_to_centres = []
        #         for centroid in centroids:
        #             edge, path = max_edge_in_guided_path_for_kmeans(X, X[id], centroid, n_neighbours=3, lookahead=3, cluster_means=centroids, labels=gt, debug=False)
        #             dist_to_centres.append(edge)
        #             path_to_centres.append(path)
        #
        #
        #         print(nk_labels[id], X[id], nk_labels[nearest_indices])
        #         for path in path_to_centres:
        #             print(len(path), np.linalg.norm(np.diff(np.array(path),axis=0),axis=1))
        #         print()
        #         # Create subplots
        #         fig, axes = plt.subplots(1, len(path_to_centres), figsize=(15, 4))
        #
        #         # Plot scatter plots on each subplot
        #         for i in range(len(path_to_centres)):
        #             if len(path_to_centres[i]) > 1:
        #                 max_edge = dist_to_centres[i]
        #                 path = path_to_centres[i]
        #
        #                 label_color = [LABEL_COLOR_MAP[l] for l in nk_labels]
        #                 axes[i].set_title(f'Distance: {max_edge:.4f}')
        #                 axes[i].scatter(X[:, 0], X[:, 1], color=label_color, marker='o', edgecolors='k')
        #                 axes[i].scatter(centroids[:, 0], centroids[:, 1], color="white", marker="X", edgecolors='k', s=200)
        #                 # axes[i].scatter(path[0][0], path[0][1], color='white', label="start", edgecolors='k', s=100)
        #                 axes[i].scatter(X[id, 0], X[id, 1], color=label_color[id], label="start", edgecolors='k', s=100)
        #                 axes[i].scatter(path[-1][0], path[-1][1], color='black', label="end", edgecolors='k', s=100)
        #
        #                 for j in range(len(path) - 1):
        #                     axes[i].plot([path[j][0], path[j + 1][0]], [path[j][1], path[j + 1][1]], color='black', linewidth=2)
        #
        #                 axes[i].legend()
        #         plt.show()
        #         plt.close()


def run_kmeans_set2():
    datasets = create_set2()

    for i_dataset, (X, gt) in enumerate(datasets):

        km = KMeans(n_clusters=2, ).fit(X)
        k_labels = km.labels_

        newKmeans = NewKMeansClustering(X, len(np.unique(gt)), neighbors=5, lookahead=20)
        nk_labels, centroids = newKmeans.fit(X)

        sc = SpectralClustering(n_clusters=len(np.unique(gt)), eigen_solver="arpack", affinity="nearest_neighbors", random_state=0).fit(X)
        s_labels = sc.labels_

        print(f"{adjusted_rand_score(k_labels, gt):.3f}, {adjusted_rand_score(nk_labels, gt):.3f}, {adjusted_rand_score(s_labels, gt):.3f}")
        print(f"{adjusted_mutual_info_score(k_labels, gt):.3f}, {adjusted_mutual_info_score(nk_labels, gt):.3f}, {adjusted_mutual_info_score(s_labels, gt):.3f}")
        print()



def analysis_kmeans_by_metrics():
    n_samples = 1000

    data1 = create_data1(n_samples)
    data2 = create_data2(n_samples)
    data3 = create_data3(n_samples)
    data4 = create_data4(n_samples)
    data5 = create_data5(n_samples)
    data6 = create_data6(n_samples)
    data7 = create_data7(n_samples)

    datasets = [
        data1,
        data2,
        data3,
        data4,
        data5,
        data6,
        data7,
    ]

    for i_dataset, (X, gt) in enumerate(datasets):
        label_color = [LABEL_COLOR_MAP[i] for i in gt]
        plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
        # plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_gt.svg")
        plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_gt.png")
        plt.close()

        Kmeans = KMeansClustering(X, len(np.unique(gt)))
        k_labels = Kmeans.fit(X)

        label_color = [LABEL_COLOR_MAP[i] for i in k_labels]
        plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
        # plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_k.svg")
        plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_k.png")
        plt.show()
        plt.close()

        newKmeans = NewKMeansClustering(X, len(np.unique(gt)), neighbors=5, lookahead=20)
        nk_labels, centroids = newKmeans.fit(X)

        label_color = [LABEL_COLOR_MAP[i] for i in nk_labels]
        plt.scatter(X[:, 0], X[:, 1], c=label_color, marker='o', edgecolors='k', alpha=0.75, s=25)
        # plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_nk.svg")
        plt.savefig(f"./figs/kmeans/data{i_dataset + 1}_nk.png")
        plt.show()
        plt.close()

        print(f"{adjusted_rand_score(k_labels, gt):.3f}, {adjusted_rand_score(nk_labels, gt):.3f}")
        print(f"{adjusted_mutual_info_score(k_labels, gt):.3f}, {adjusted_mutual_info_score(nk_labels, gt):.3f}")
        print()





def analyze_time_kmeans_examples():
    for n_samples in [100, 500, 1000, 5000, 10000]:
        X, gt = create_data3(n_samples)

        # k_start = time.time()
        # Kmeans = KMeansClustering(X, len(np.unique(gt)))
        # k_labels = Kmeans.fit(X)
        # k_time = time.time() - k_start
        #
        # nk_start = time.time()
        # newKmeans = NewKMeansClustering(X, len(np.unique(gt)), neighbors=5, lookahead=20)
        # nk_labels, centroids = newKmeans.fit(X)
        # nk_time = time.time() - k_start

        s_start = time.time()
        sc = SpectralClustering(n_clusters=len(np.unique(gt)), eigen_solver="arpack", affinity="nearest_neighbors", random_state=0).fit(X)
        s_labels = sc.labels_
        s_time = time.time() - s_start

        print(f"D3 with {n_samples}samples, "
              # f"KMeans ({adjusted_rand_score(k_labels, gt):.3f}/{adjusted_mutual_info_score(k_labels, gt):.3f}): {k_time:.3f}s, "
              # f"NewKMeans ({adjusted_rand_score(nk_labels, gt):.3f}/{adjusted_mutual_info_score(nk_labels, gt):.3f}): {nk_time:.3f}s"
              f"SpectralClustering ({adjusted_rand_score(s_labels, gt):.3f}/{adjusted_mutual_info_score(s_labels, gt):.3f}): {s_time:.3f}s"
              )



    # for n_samples in [5000, 10000]:
    #     X, gt = create_data3(n_samples)
    #
    #     k_start = time.time()
    #     Kmeans = KMeansClustering(X, len(np.unique(gt)))
    #     k_labels = Kmeans.fit(X)
    #     k_time = time.time() - k_start
    #
    #     nk_start = time.time()
    #     newKmeans = NewKMeansClustering(X, len(np.unique(gt)), neighbors=50, lookahead=20)
    #     nk_labels, centroids = newKmeans.fit(X)
    #     nk_time = time.time() - k_start
    #
    #     print(f"D3 with {n_samples}samples, "
    #           f"KMeans ({adjusted_rand_score(k_labels, gt):.3f}/{adjusted_mutual_info_score(k_labels, gt):.3f}): {k_time:.3f}s, "
    #           f"NewKMeans ({adjusted_rand_score(nk_labels, gt):.3f}/{adjusted_mutual_info_score(nk_labels, gt):.3f}): {nk_time:.3f}s")


def analyze_time_kmeans_features():
    for n_features in [2,3,4,5,6]:
        n_samples = 1000
        X, gt = create_data3(n_samples, n_features)

        # k_start = time.time()
        # Kmeans = KMeansClustering(X, len(np.unique(gt)))
        # k_labels = Kmeans.fit(X)
        # k_time = time.time() - k_start
        #
        # nk_start = time.time()
        # newKmeans = NewKMeansClustering(X, len(np.unique(gt)), neighbors=5, lookahead=20)
        # nk_labels, centroids = newKmeans.fit(X)
        # nk_time = time.time() - k_start

        s_start = time.time()
        sc = SpectralClustering(n_clusters=len(np.unique(gt)), eigen_solver="arpack", affinity="nearest_neighbors", random_state=0).fit(X)
        s_labels = sc.labels_
        s_time = time.time() - s_start

        print(f"D3 with {n_samples}samples, "
              # f"KMeans ({adjusted_rand_score(k_labels, gt):.3f}/{adjusted_mutual_info_score(k_labels, gt):.3f}): {k_time:.3f}s, "
              # f"NewKMeans ({adjusted_rand_score(nk_labels, gt):.3f}/{adjusted_mutual_info_score(nk_labels, gt):.3f}): {nk_time:.3f}s"
              f"SpectralClustering ({adjusted_rand_score(s_labels, gt):.3f}/{adjusted_mutual_info_score(s_labels, gt):.3f}): {s_time:.3f}s"
              )



if __name__ == '__main__':
    # run_scores_set1()
    run_scores_set2()
    # run_scores_set3d()

    # analyze_time_scores_examples()
    # analyze_time_scores_features()

    # run_kmeans_set1()
    # run_kmeans_set2()

    # analyze_time_kmeans_examples()
    # analyze_time_kmeans_features()