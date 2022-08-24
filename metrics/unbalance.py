from typing import Callable, Dict, Optional

import numpy as np


def get_binary_cm(ln: float, lp: float, delta: float) -> np.array:
    """Construct a binary confusion matrix from the "negative" and "positive" sensitivities and a balance
    parameter delta so that delta=0 is balanced, and -1 <= delta <= 1."""
    cm = np.zeros((2, 2))
    cm[0, 0] = ln * (1 - delta) / 2
    cm[1, 0] = (1 - lp) * (1 + delta) / 2
    cm[0, 1] = (1 - ln) * (1 - delta) / 2
    cm[1, 1] = lp * (1 + delta) / 2
    return cm


def rescale_metric(func: Callable, cm: np.array) -> float:
    """Rescale metrics that are in the range [-1, 1] so that they are in the range [0, 1]"""
    return (func(cm)+1)/2


def perclass(func: Callable, cm: np.array, c: int) -> float:
    """Helper function that can be used with functools.partial to get a callable such as :

    F1p = partial(perclass, F1c, c=1)

    Which will be directly usable as : F1p(cm), calling F1c(cm, c=1)"""
    return func(cm, c)


def get_binary_metrics_results(metrics: Dict[str, Callable],
                               deltas: np.array,
                               lns: np.array,
                               lps: np.array) -> np.array:
    """Get the results for all metrics sampling the delta/lns/lps space."""
    all_results = {}
    for key in metrics:
        all_results[key] = np.zeros((len(deltas), len(lns), len(lps)))

    for i_d, delta in enumerate(deltas):
        for i_n, ln in enumerate(lns):
            for i_p, lp in enumerate(lps):
                cm = get_binary_cm(ln, lp, delta)
                for key, metric in metrics.items():
                    all_results[key][i_d, i_n, i_p] = metric(cm)

    return all_results


def _beta_fn(beta: float, m: int) -> float:
    """Get unbalance factor based on unbalance parameters and number of classes"""
    return beta + (1-beta)/m


def get_distributed_proportions(m: int, beta: float) -> np.array:
    """Get class proportions based on a distributed unbalance"""
    pis = np.zeros((m,))
    betas = np.array([_beta_fn(beta, m-i) for i in np.arange(m)])
    pis[0] = betas[0]
    for i in range(1, m):
        pis[i] = np.prod(1-betas[:i])*betas[i]
    return pis


def error_matrix(m: int, seed: int = 0) -> np.array:
    """Get random error-distribution matrix"""
    np.random.seed(seed)
    em = np.random.random((m, m))
    for i in range(m):
        em[i, i] = 0
        em[i, :] = em[i, :] / em[i, :].sum()

    return em


def cm_multiclass_ed(m: int, lambdas: np.array, beta: float) -> np.array:
    """ED scenario"""
    pis = get_distributed_proportions(m, beta)
    cm = np.zeros((m, m))
    for i in range(m):
        cm[i, i] = pis[i] * lambdas[i]
        for j in range(m):
            if i == j:
                continue
            cm[i, j] = pis[i] * (1 - lambdas[i]) / (m - 1)

    return cm


def cm_multiclass_rd(m: int, lambdas: np.array, beta: float, em: Optional[np.array] = None) -> np.array:
    """RD scenario"""
    if em is None:
        em = error_matrix(m)

    pis = get_distributed_proportions(m, beta)
    cm = np.zeros((m, m))

    for i in range(m):
        cm[i, i] = pis[i] * lambdas[i]
        for j in range(m):
            if i == j:
                continue
            cm[i, j] = pis[i] * em[i, j] * (1 - lambdas[i])

    return cm


def cm_multiclass_bo(m, lambdas, beta):
    """BO scenario"""
    pis = get_distributed_proportions(m, beta)
    cm = np.zeros((m, m))

    for i in range(m):
        cm[i, i] = pis[i] * lambdas[i]
        for j in range(m):
            if i == j:
                continue
            cm[i, j] = pis[i] * (pis[j] / (pis.sum() - pis[i])) * (1 - lambdas[i])

    return cm
