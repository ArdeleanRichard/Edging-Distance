import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

from datasets import create_data7
from distance import max_edge_in_guided_path2

def create_demo_data(n_samples=100):
    avgPoints = n_samples // 2
    C1 = [-1, -1] + .1 * np.random.randn(avgPoints, 2)
    C2 = [1, 1] + .1 * np.random.randn(avgPoints, 2)

    X = np.vstack((C1, C2))

    c1Labels = np.full(len(C1), 0)
    c2Labels = np.full(len(C2), 1)

    y = np.hstack((c1Labels, c2Labels))

    data = (X, y)

    return data


def plot_edging_distance(title, X, max_edge, path):
    plt.title(f'{title}: {max_edge:.4f}')
    plt.scatter(X[:, 0], X[:, 1], color='red', marker='o', edgecolors='k')

    for i in range(len(path) - 1):
        # plt.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], color='black', linewidth=2, label='path')
        plt.annotate('', xy=(path[i + 1][0], path[i + 1][1]), xytext=(path[i][0], path[i][1]), arrowprops=dict(facecolor='black', width=1, headlength=10, headwidth=6), zorder=1)

    plt.scatter(path[0][0], path[0][1], color='white', label="start", edgecolors='k', s=100, alpha=1)
    plt.scatter(path[-1][0], path[-1][1], color='black', label="end", edgecolors='k', s=100, alpha=1)

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()
    plt.close()

def plot_euclidean_distance(title, X, start, end):
    max_edge = np.linalg.norm(start - end)
    plt.title(f'{title}: {max_edge:.4f}')
    plt.scatter(X[:, 0], X[:, 1], color='red', marker='o', edgecolors='k')

    path = [start, end]
    for i in range(len(path) - 1):
        # plt.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], color='black', linewidth=2, label='path')
        plt.annotate('', xy=(path[i + 1][0], path[i + 1][1]), xytext=(path[i][0], path[i][1]), arrowprops=dict(facecolor='black', width=1, headlength=10, headwidth=6), zorder=1)

    plt.scatter(path[0][0], path[0][1], color='white', label="start", edgecolors='k', s=100, alpha=1)
    plt.scatter(path[-1][0], path[-1][1], color='black', label="end", edgecolors='k', s=100, alpha=1)

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()
    plt.close()



def demo_path():
    X, y = create_demo_data()
    start = X[0]
    end_intra = X[49]
    end_inter = X[99]
    intra_edge, intra_path = max_edge_in_guided_path2(X, start, end_intra, n_neighbours=5, lookahead=3)
    inter_edge, inter_path = max_edge_in_guided_path2(X, start, end_inter, n_neighbours=5, lookahead=3)

    plot_edging_distance("IntraCluster Edging Distance", X, intra_edge, intra_path)
    plot_edging_distance("InterCluster Edging Distance", X, inter_edge, inter_path)

    plot_euclidean_distance("IntraCluster Euclidean Distance", X, start, end_intra)
    plot_euclidean_distance("InterCluster Euclidean Distance", X, start, end_inter)


def demo_circles():
    X, y = datasets.make_circles(n_samples=1000, shuffle=False, factor=0.5, noise=0.05, random_state=42)
    start = X[0]
    end_intra = X[250]
    end_inter = X[777]
    intra_edge, intra_path = max_edge_in_guided_path2(X, start, end_intra, n_neighbours=5, lookahead=3)
    inter_edge, inter_path = max_edge_in_guided_path2(X, start, end_inter, n_neighbours=5, lookahead=3)

    plot_edging_distance("IntraCluster Edging Distance", X, intra_edge, intra_path)
    plot_edging_distance("InterCluster Edging Distance", X, inter_edge, inter_path)

    plot_euclidean_distance("IntraCluster Euclidean Distance", X, start, end_intra)
    plot_euclidean_distance("InterCluster Euclidean Distance", X, start, end_inter)


if __name__ == '__main__':
    # demo_path()
    demo_circles()
