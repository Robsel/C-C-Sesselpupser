import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs


def create_samples():
    global X
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)


create_samples()

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)


def do_meanshift():
    global labels, cluster_centers, n_clusters_
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)


do_meanshift()


def plot():
    plt.figure(1)
    plt.clf()
    colors = ["#dede00", "#377eb8", "#f781bf"]
    markers = ["x", "o", "^"]
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], markers[k], color=col)
        plt.plot(
            cluster_center[0],
            cluster_center[1],
            markers[k],
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=14,
        )
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()


plot()
