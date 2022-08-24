from typing import Tuple, Optional

import numpy as np
from sklearn.metrics import auc


def F1c(conf_mat: np.array, c: int) -> Optional[float]:
    """Per-class F1-score of class c"""
    TP = conf_mat[c, c]
    FP = conf_mat[:, c].sum() - TP
    FN = conf_mat[c, :].sum() - TP
    if 2*TP+FP+FN == 0:
        return None
    return 2 * TP / (2 * TP + FP + FN)


def PrecRec(conf_mat: np.array, c: int) -> Tuple[Optional[float], Optional[float]]:
    """Per-class precision & recall of class c"""
    TP = conf_mat[c, c]
    FP = conf_mat[:, c].sum() - TP
    FN = conf_mat[c, :].sum() - TP
    if TP + FP == 0:
        return 0, TP / (TP+FN)
    return TP / (TP + FP), TP / (TP + FN)


def multiclass_cm(gt: np.array, pred: np.array) -> np.array:
    """Compute multi-class CM from gt labels & prediction vector."""
    pred_label = pred.argmax(axis=1)
    cm = np.zeros((pred.shape[1], pred.shape[1]))
    for g, p in zip(gt, pred_label):
        cm[g, p] += 1
    return cm


def binary_cm(gt: np.array, pred: np.array, c: int, tau: float = 0.5) -> np.array:
    """Get binary CM for a given class & a confidence level"""
    cm = np.zeros((2, 2))
    for g, p in zip(gt, pred[:, c]):
        cm[int(g == c), int(p >= tau)] += 1
    return cm


def SENc(conf_mat: np.array, c: int = 1) -> float:
    TP = conf_mat[c, c]
    FN = conf_mat[c, :].sum()-TP
    return TP/(TP+FN)


def SPEc(conf_mat: np.array, c: int = 1) -> float:
    TP = conf_mat[c, c]
    FP = conf_mat[:, c].sum()-TP
    FN = conf_mat[c, :].sum()-TP
    TN = conf_mat.sum()-TP-FP-FN
    return TN/(TN+FP)


def get_roc(gt: np.array, pred: np.array, c: int) -> np.array:
    roc = []
    for t in np.arange(0, 1.01, 0.01):
        cm = binary_cm(gt, pred, c, t)
        roc.append([SENc(cm), 1-SPEc(cm)])

    return np.array(roc)


def get_micro_roc(gt: np.array, pred: np.array) -> np.array:
    roc = []
    for idt, t in enumerate(np.arange(0, 1.01, 0.01)):
        cm_micro = np.zeros((2, 2))
        for c in range(pred.shape[1]):
            cm_micro += binary_cm(gt, pred, c, t)
        roc.append((SENc(cm_micro), 1-SPEc(cm_micro)))
    return np.array(roc)


def auroc(roc: np.array) -> float:
    return auc(roc[:, 1], roc[:, 0])


def get_micro_auroc(gt: np.array, pred: np.array) -> float:
    roc = get_micro_roc(gt, pred)

    return auroc(roc)


def get_macro_auroc(gt: np.array, pred: np.array) -> float:
    aurocs = []
    for c in range(pred.shape[1]):
        roc = get_roc(gt, pred, c)
        aurocs.append(auc(roc[:, 1], roc[:, 0]))
    return sum(aurocs)/len(aurocs)


def GM(conf_mat: np.array) -> float:
    """Geometric Mean of per-class SEN"""
    result = 1
    n_c = conf_mat.shape[0]
    for c in range(n_c):
        TP = conf_mat[c, c]
        FN = conf_mat[c, :].sum() - TP
        result *= TP/(TP+FN)
    return result**(1/n_c)


def Accuracy(conf_mat: np.array) -> float:
    """Overall accuracy"""
    return np.trace(conf_mat).sum() / conf_mat.sum()


def MacroF1(conf_mat: np.array, harmonic=True) -> float:
    """Macro-averaged F1 score"""
    if harmonic:
        return hF1(conf_mat)
    return sF1(conf_mat)


def sF1(conf_mat: np.array) -> float:
    """Macro-F1 using "simple macro-averaging"."""
    return 0.5*(F1c(conf_mat, 0) + F1c(conf_mat, 1))


def hF1(conf_mat: np.array) -> float:
    """Macro-F1 using "harmonic macro-averaging"."""
    Ps = []
    Rs = []
    for c in range(conf_mat.shape[0]):
        if conf_mat[c].sum() == 0:
            continue
        P, R = PrecRec(conf_mat, c)
        Ps.append(P)
        Rs.append(R)
    MP = np.mean(Ps)
    MR = np.mean(Rs)
    return 2 * MP * MR / (MP + MR)


def wKappa(conf_mat: np.array, w: np.array) -> float:
    """Weighted kappa, defined as:
    k = 1 - (sum_i sum_j w_ij CM_ij)/(sum_i sum_j w_ij e_ij)
    """
    e = np.zeros_like(conf_mat, dtype=np.float64)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[0]):
            e[i, j] = conf_mat[i, :].sum() * conf_mat[:, j].sum() / conf_mat.sum()
    return 1 - (w * conf_mat).sum() / (w * e).sum()


def kappaU(conf_mat: np.array) -> float:
    """Unweighted kappa -> w_ij = 1 - delta_ij"""
    w = 1 - np.eye(conf_mat.shape[0])
    return wKappa(conf_mat, w)


def kappaL(conf_mat: np.array) -> float:
    """Linear weighted kappa -> w_ij = |i-j|"""
    wL = np.zeros(conf_mat.shape)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            wL[i, j] = np.abs(i - j)
    return wKappa(conf_mat, wL)


def kappaQ(conf_mat: np.array) -> float:
    """Quadratic weighted kappa -> w_ij = (i-j)**2"""
    wQ = np.zeros(conf_mat.shape)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            wQ[i, j] = (i - j) ** 2
    return wKappa(conf_mat, wQ)


def MCC(conf_mat: np.array) -> float:
    """Multi-class Matthews Correlation Coefficient"""
    c = conf_mat.shape[0]  # #classes
    num = conf_mat.sum() * np.trace(conf_mat).sum()
    for k in range(c):
        num -= conf_mat[:, k].sum() * conf_mat[k, :].sum()

    denum1 = denum2 = conf_mat.sum() ** 2
    for k in range(c):
        denum1 -= conf_mat[k, :].sum() ** 2
        denum2 -= conf_mat[:, k].sum() ** 2
    denum1 = np.sqrt(denum1)
    denum2 = np.sqrt(denum2)
    return num / (denum1 * denum2)
