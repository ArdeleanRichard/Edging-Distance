from copy import copy

import numpy as np
from matplotlib import pyplot as plt

from distance import k_nearest_neighbors, max_edge_in_guided_path2


def plot(X, max_edge, path):
    plt.title(f'Distance: {max_edge:.4f}')
    plt.scatter(X[:, 0], X[:, 1], color='red', marker='o', edgecolors='k')


    for i in range(len(path) - 1):
        # plt.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], color='black', linewidth=2, label='path')
        plt.annotate('', xy=(path[i + 1][0], path[i + 1][1]), xytext=(path[i][0], path[i][1]), arrowprops=dict(facecolor='black', width=1, headlength=10, headwidth=6), zorder=-1)

    plt.scatter(path[0][0], path[0][1], color='white', label="start", edgecolors='k', s=100, alpha=1)
    plt.scatter(path[-1][0], path[-1][1], color='black', label="end", edgecolors='k', s=100, alpha=1)

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()
    plt.close()




# Non-negativity: The distance between two objects is always non-negative:
# d(x,y)≥0 where d(x,y) is the distance between objects x and y, and equality holds if and only if x=y.
#
# Identity of indiscernibles: The distance between an object and itself is zero:
# d(x,x)=0
#
# Symmetry: The distance from object x to object y is the same as the distance from
# d(x,y)=d(y,x)
#
# Triangle inequality: The distance from one object to another through a third object is less than or equal to the sum of the distances directly from one object to the other two:
# d(x,z)≤d(x,y)+d(y,z) for all objects

def demonstrate_non_negativity():
    # Example usage:
    # Create some points dataset
    X = np.array([(-1, -1), (0, 0), (1, 1)])

    # Define start and end points
    start = np.array((0, 0))
    end = np.array((1, 1))

    # Call the function
    max_edge, path = max_edge_in_guided_path2(X, start, end, n_neighbours=2)
    plot(X, max_edge, path)


    # Define start and end points
    start = np.array((0, 0))
    end = np.array((-1, -1))

    max_edge, path = max_edge_in_guided_path2(X, start, end, n_neighbours=2)
    plot(X, max_edge, path)


def demonstrate_identity():
    # Example usage:
    # Create some points dataset
    X = np.array([(-1, -1), (0, 0), (1, 1)])

    # Define start and end points
    start = np.array((0, 0))
    end = np.array((0, 0))

    # Call the function
    max_edge, path = max_edge_in_guided_path2(X, start, end)
    plot(X, max_edge, path)


def demonstrate_symmetry():
    # Example usage:
    # Create some points dataset
    X = np.array([(0, 0), (1, 1), (2,2), (3,3),(4,4)])

    # Define start and end points
    start = np.array((0, 0))
    end = np.array((4, 4))

    # Call the function
    max_edge, path = max_edge_in_guided_path2(X, start, end, n_neighbours=1)
    plot(X, max_edge, path)

    # Define start and end points
    start = np.array((4, 4))
    end = np.array((0, 0))

    max_edge, path = max_edge_in_guided_path2(X, start, end, n_neighbours=1)
    plot(X, max_edge, path)

def demonstrate_triangle_inequality():
    # Example usage:
    # Create some points dataset
    X = np.array([(0, 0), (1, 1), (2,2)])

    # Define start and end points
    start = np.array((0, 0))
    end = np.array((1, 1))

    # Call the function
    max_edge, path = max_edge_in_guided_path2(X, start, end, n_neighbours=2)
    plot(X, max_edge, path)

    # Define start and end points
    start = np.array((1, 1))
    end = np.array((2, 2))

    max_edge, path = max_edge_in_guided_path2(X, start, end, n_neighbours=2)
    plot(X, max_edge, path)

    # Define start and end points
    start = np.array((0, 0))
    end = np.array((2, 2))

    # Call the function
    max_edge, path = max_edge_in_guided_path2(X, start, end, n_neighbours=1)
    plot(X, max_edge, path)


if __name__ == '__main__':
    demonstrate_non_negativity()
    demonstrate_identity()
    demonstrate_symmetry()
    demonstrate_triangle_inequality()