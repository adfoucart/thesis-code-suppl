from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import numpy as np


def compute_and_plot_mds(diss, labels, seed=None):
    mds = MDS(dissimilarity='precomputed', random_state=seed)

    coords = mds.fit_transform(diss)
    dcoords = np.zeros_like(diss)
    for i in range(diss.shape[0]):
        for j in range(diss.shape[1]):
            dcoords[i, j] = np.abs(np.sqrt(((coords[i] - coords[j]) ** 2).sum()))
    err = np.abs(dcoords - diss).sum(axis=0) / dcoords.shape[0]

    fig, ax = plt.subplots(figsize=(15, 15))
    for i in range(diss.shape[0]):
        plt.plot(coords[i, 0], coords[i, 1], 'o')
        circle = plt.Circle(coords[i, :], err[i], fill=False)
        ax.add_artist(circle)
        plt.text(coords[i, 0] + 0.01, coords[i, 1] + 0.01, labels[i])
    ax.set_aspect('equal')
    plt.show()
