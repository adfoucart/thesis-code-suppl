from typing import Callable, Optional

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt
from skimage.metrics import hausdorff_distance
from .classification import MCC


def IoU(a: np.array, b: np.array, none_value: Optional[float] = None) -> Optional[float]:
    """Computes the IoU between the binary masks a and b"""
    if a.dtype != bool:
        a = a > 0
    if b.dtype != bool:
        b = b > 0

    if (a+b).sum() == 0:
        return none_value
    return (a * b).sum() / (a + b).sum()


def IoU_cm(a: np.array,
           b: np.array,
           weight_func: Callable[[np.array, int], np.array] = None,
           n: int = 1,
           none_value: Optional[float] = None) -> Optional[float]:
    """Computes the IoU based on the confusion matrix, with the possibility of using a weighted confusion matrix."""
    if a.dtype != bool:
        a = a > 0
    if b.dtype != bool:
        b = b > 0

    if weight_func is None:
        cm = get_cm(a, b)
    else:
        cm = get_weighted_cm(a, b, n, weight_func)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    if tp + fp + fn == 0:
        return none_value
    return tp / (tp + fp + fn)


def DSC(a: np.array, b: np.array, none_value: Optional[float] = None) -> Optional[float]:
    """Computes the DSC between the binary masks a and b"""
    if a.dtype != bool:
        a = a > 0
    if b.dtype != bool:
        b = b > 0

    if a.sum() + b.sum() == 0:
        return none_value
    return 2*(a * b).sum() / (a.sum() + b.sum())


def DSC_cm(a: np.array,
           b: np.array,
           weight_func: Callable[[np.array, int], np.array] = None,
           n: int = 1,
           none_value: Optional[float] = None) -> Optional[float]:
    """Computes the DSC based on the confusion matrix, with the possibility of using a weighted confusion matrix."""
    if a.dtype != bool:
        a = a > 0
    if b.dtype != bool:
        b = b > 0
    if weight_func is None:
        cm = get_cm(a, b)
    else:
        cm = get_weighted_cm(a, b, n, weight_func)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    if 2*tp + fp + fn == 0:
        return none_value
    return 2*tp / (2*tp + fp + fn)


def uaIoU(gt: np.array, pred: np.array, tau: int) -> float:
    """Uncertainty-aware IoU"""
    if gt.dtype != bool:
        gt = gt > 0
    if pred.dtype != bool:
        pred = pred > 0

    gt_plus = edt(gt == 0) <= tau
    gt_minus = edt(gt) > tau

    TP = (gt_plus * pred).sum()
    FP = ((1 - gt_plus) * pred).sum()
    FN = (gt_minus * (1 - pred)).sum()

    return TP / (TP + FP + FN)


def uaDSC(gt: np.array, pred: np.array, tau: int) -> float:
    """Uncertainty-aware DSC"""
    if gt.dtype != bool:
        gt = gt > 0
    if pred.dtype != bool:
        pred = pred > 0

    gt_plus = edt(gt == 0) <= tau
    gt_minus = edt(gt) > tau

    TP = (gt_plus * pred).sum()
    FP = ((1 - gt_plus) * pred).sum()
    FN = (gt_minus * (1 - pred)).sum()

    return 2 * TP / (2 * TP + FP + FN)


def get_cm(a, b):
    """Computes the confusion matrix between the binary masks a and be."""
    if a.dtype != bool:
        a = a > 0
    if b.dtype != bool:
        b = b > 0

    cm = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            cm[i, j] = ((a == i) * (b == j)).sum()

    return cm


def get_fuzzy_weights(a: np.array, n: int = 1) -> np.array:
    """Changes the binary mask a so that the border area has less weight than the clear fg / bg area"""
    ref_fuzzy = np.ones_like(a).astype('float')

    edt_bg = edt(a == 0)
    edt_fg = edt(a)

    ref_fuzzy[edt_bg > n] = 1
    ref_fuzzy[(edt_bg <= n) * (edt_bg > 0)] = edt_bg[(edt_bg <= n) * (edt_bg > 0)] / n
    ref_fuzzy[(edt_fg <= n) * (edt_fg > 0)] = edt_fg[(edt_fg <= n) * (edt_fg > 0)] / n
    ref_fuzzy[edt_fg > n] = 1

    return ref_fuzzy


def get_removed_weights(a: np.array, n: int = 1) -> np.array:
    """Get a region of n pixels around the borders of the binary mask a"""
    edt_bg = edt(a == 0)
    edt_fg = edt(a)
    fuzzy_area = (edt_bg <= n) * (edt_bg > 0) + (edt_fg <= n) * (edt_fg > 0)

    return 1-fuzzy_area


def get_weighted_cm(a: np.array, b: np.array, n: int = 1, weight_func: Callable = get_fuzzy_weights):
    """Fuzzy CM based on weighting all the pixels in the uncertain region less."""
    if a.dtype != bool:
        a = a > 0
    if b.dtype != bool:
        b = b > 0

    weights = weight_func(a, n)

    tp = (a * b * weights).sum()
    tn = ((1 - a) * (1 - b) * weights).sum()
    fp = ((1 - a) * b * weights).sum()
    fn = (a * (1 - b) * weights).sum()

    cm_fuzz = np.array([
        [tn, fp],
        [fn, tp]
    ])

    return cm_fuzz


def HD(a: np.array, b: np.array, inner: bool = True, outer: bool = False, percentile: Optional[float] = None) -> float:
    """Computes Hausdorff's distance between the contours of binary masks a and b
    (assumed to contain contiguous pixels). We can specify if we use outer contours, inner contours, or both."""
    if a.dtype != bool:
        a = a > 0
    if b.dtype != bool:
        b = b > 0

    c_a = (edt(a) == 1)*inner + (edt(a == 0) == 1)*outer
    c_b = (edt(b) == 1)*inner + (edt(b == 0) == 1)*outer

    if percentile is not None:
        return HD_custom(a, b, inner, outer, percentile)

    return hausdorff_distance(c_a, c_b)


def HD_custom(a: np.array,
              b: np.array,
              inner: bool = True,
              outer: bool = False,
              percentile: Optional[float] = None,
              tau: float = 1.) -> float:
    """Computes Hausdorff's distance between the contours of binary masks a and b (assumed to contain contiguous
    pixels). We can specify if we use outer contours, inner contours, or both, and a HD percentile."""
    if a.dtype != bool:
        a = a > 0
    if b.dtype != bool:
        b = b > 0

    c_a = (edt(a) == 1) * inner  # + (edt(a == 0) == 1)*outer
    c_b = (edt(b) == 1) * inner  # + (edt(b == 0) == 1)*outer

    c_a_t = (edt(a) <= tau) * inner * a + (edt(a == 0) <= tau) * outer * (a == 0)
    c_b_t = (edt(b) <= tau) * inner * b + (edt(b == 0) <= tau) * outer * (b == 0)

    dc_a = edt(c_a_t == 0)[c_b]
    dc_b = edt(c_b_t == 0)[c_a]

    if percentile is None:
        return max(dc_a.max(), dc_b.max())

    n_a = int(percentile*len(dc_a))
    n_b = int(percentile*len(dc_b))

    return max(np.sort(dc_a.flatten())[n_a], np.sort(dc_b.flatten())[n_b])


def MCC_segmentation(a: np.array, b: np.array) -> float:
    cm = np.zeros((2, 2))
    cm[1, 1] = (a * b).sum()
    cm[1, 0] = (a * (b == False)).sum()
    cm[0, 1] = ((a == False) * b).sum()
    cm[0, 0] = ((a == False) * (b == False)).sum()
    return MCC(cm)
