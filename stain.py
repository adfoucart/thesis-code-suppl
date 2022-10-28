import numpy as np

"""
Stain normalisation adapted from:
- https://github.com/schaugf/HEnorm_python
"""

HE_REF = np.array([
    [0.5626, 0.2159],
    [0.7201, 0.8012],
    [0.4062, 0.5581]])
MAX_C_REF = np.array([1.9705, 1.0308])
IO = 240


def rgb2od(im: np.array) -> np.array:
    """Convert RGB intensity to RGB Optical Density"""
    return -np.log((im.astype(np.float) + 1) / IO)


def od2hevec(od: np.array, alpha=1, beta=0.15) -> np.array:
    """Find HE vectors from Optical Density image"""

    # remove transparent pixels
    ODhat = od[~np.any(od < beta, axis=2)]

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    return HE


def od2hec(od: np.array, hevec: np.array) -> np.array:
    """Converts RGB Optical Density image to HE concentration images"""
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(od, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(hevec, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, MAX_C_REF)
    C2 = np.divide(C, tmp[:, np.newaxis]).T

    return C2.reshape((od.shape[0], od.shape[1], -1))


def rgb2hec(im: np.array) -> np.array:
    od = rgb2od(im)
    hevec = od2hevec(od)
    return od2hec(od, hevec)


def hec2norm(hec: np.array):
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(IO, np.exp(-HE_REF.dot(hec.reshape((-1, 2)).T)))
    Inorm[Inorm > 255] = 254
    return np.reshape(Inorm.T, (hec.shape[0], hec.shape[1], 3)).astype(np.uint8)


def rgb2norm(im: np.array) -> np.array:
    """Normalize the (H&E) stain appearance of the RGB image."""
    hec = rgb2hec(im)
    return hec2norm(hec)


def hec2unmixed(hec: np.array):
    """Get a RGB visualisation of the H&E channel concentrations using their reference colors"""
    # unmix hematoxylin and eosin
    hc = hec[..., 0].flatten()
    ec = hec[..., 1].flatten()
    H = np.multiply(IO, np.exp(np.expand_dims(-HE_REF[:, 0], axis=1).dot(np.expand_dims(hc, axis=0))))
    H[H > 255] = 255
    H = np.reshape(H.T, (hec.shape[0], hec.shape[1], 3)).astype(np.uint8)
    E = np.multiply(IO, np.exp(np.expand_dims(-HE_REF[:, 1], axis=1).dot(np.expand_dims(ec, axis=0))))
    E[E > 255] = 255
    E = np.reshape(E.T, (hec.shape[0], hec.shape[1], 3)).astype(np.uint8)
    return H, E


def normalizeStaining(img, alpha=1, beta=0.15):
    """Normalize staining appearence of H&E stained images

    Example use:
        see test.py

    Input:
        img: RGB input image
        alpha:
        beta:

    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    """
    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(np.float) + 1) / IO)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, MAX_C_REF)
    C2 = np.divide(C, tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(IO, np.exp(-HE_REF.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # unmix hematoxylin and eosin
    H = np.multiply(IO, np.exp(np.expand_dims(-HE_REF[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(IO, np.exp(np.expand_dims(-HE_REF[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    return Inorm, H, E
