import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from skimage.color import rgb2hsv
from skimage.io import imread
from sklearn.decomposition import PCA

"""
@author Adrien Foucart

Methods for plotting color space information
"""


def plot_3d_rgb(im: np.array, show: bool = True) -> None:
    if im.dtype == 'uint8' or (im.dtype == 'int' and im.max() < 256):
        im = im/255
    elif im.dtype == 'uint16' or (im.dtype == 'int' and im.max() >= 256):
        im = im/65535

    rs = im[..., 0].flatten()
    gs = im[..., 1].flatten()
    bs = im[..., 2].flatten()

    cs = []
    for r, g, b in zip(rs,gs,bs):
        cs.append([r, g, b])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(rs, gs, bs, c=cs)

    if show:
        plt.show()


def plot_pca_2c(im: np.array, show: bool = True) -> None:
    im_ = im[..., :3].reshape((im.shape[0]*im.shape[1], 3))
    pca = PCA(n_components=2)
    new_coords = pca.fit_transform(im_)

    plt.figure()
    plt.scatter(new_coords[:, 0], new_coords[:, 1])
    if show:
        plt.show()


def plot_hue(im: np.array, show: bool = True) -> None:
    hsv = rgb2hsv(im[..., :3])
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.subplot(1, 2, 2)
    plt.hist(hsv[..., 0].flatten(), bins=360)
    plt.xlabel('Hue')
    plt.ylabel('N pixels')
    if show:
        plt.show()


if __name__ == "__main__":
    monusag_im = imread("D:/Adrien/dataset/MoNuSAC/MoNuSAC Testing Data and Annotations/TCGA-2Z-A9JG-01Z-00-DX1/TCGA-2Z-A9JG-01Z-00-DX1_4.tif")
    etretat_im = imread("D:/Adrien/pCloud/ULB/TPs/INFOH500/2021-2022/etretat.jpg")

    # plot_3d_rgb(monusag_im[::5, ::5])
    # plot_3d_rgb(etretat_im[::5, ::5])

    # plot_pca_2c(monusag_im, show=False)
    # plot_pca_2c(etretat_im)

    plot_hue(monusag_im, show=False)
    plot_hue(etretat_im)
