from typing import List, Tuple

import numpy as np
from sklearn.metrics import cohen_kappa_score


def epstein(p1: int, p2: int, n1: int, n2: int) -> int:
    """Epstein groups from the two most prominent Gleason patterns p1 and p2
    and the number of pixels they cover n1 and n2"""
    if n2 > 0:  # >=2 types of glands present:
        if p1 + p2 <= 6:
            return 1
        elif p1 == 3 and p2 == 4:
            return 2
        elif p1 == 4 and p2 == 3:
            return 3
        elif p1 == 4 and p2 == 4:
            return 4
        elif p1 == 5 or p2 == 5:
            return 5
        else:
            raise ValueError(f"Couldn't compute Epstein Groups: {p1}, {p2}, {n1}, {n2}")
    else:  # only 1 type
        if p1 <= 3:
            return 1
        elif p1 == 4:
            return 4
        elif p1 == 5:
            return 5
        else:
            raise ValueError(f"Couldn't compute Epstein Groups: {p1}, {p2}, {n1}, {n2}")


def get_G_SP(px_per_grades: np.array) -> np.array:
    """Simple pixel count, Gleason scores.
    px_per_grades contains the number of pixels for each pattern grade in each image in a set of annotators.
    Returns an array with the computed score per image"""
    scores = np.zeros((px_per_grades.shape[0], px_per_grades.shape[1])).astype('int')
    for idi in range(px_per_grades.shape[0]):
        for idm in range(px_per_grades.shape[1]):
            if px_per_grades[idi, idm].sum() > 0:
                h = px_per_grades[idi, idm][2:]  # only look at grades 3-4-5
                if h.sum() == 0:
                    scores[idi, idm] = 0
                else:
                    s = np.argsort(h)[::-1]
                    p1 = s[0]
                    p2 = s[1]

                    if h[p2] > 0:  # 2 types of glands present:
                        scores[idi, idm] = (p1 + 2) + (p2 + 2)
                    else:  # only 1 type
                        scores[idi, idm] = (p1 + 2) * 2

    return scores


def get_E_SP(px_per_grades: np.array) -> np.array:
    """Simple pixel count, Epstein groups"""
    scores = np.ones((px_per_grades.shape[0], px_per_grades.shape[1])).astype('int')
    for idi in range(px_per_grades.shape[0]):
        for idm in range(px_per_grades.shape[1]):
            if px_per_grades[idi, idm].sum() > 0:
                h = px_per_grades[idi, idm][2:]
                if h.sum() == 0:
                    scores[idi, idm] = 1
                else:
                    s = np.argsort(h)[::-1]
                    p1 = s[0]
                    p2 = s[1]
                    scores[idi, idm] = epstein(p1 + 2, p2 + 2, h[p1], h[p2])

    return scores


def get_G_ISUP(px_per_grades: np.array) -> np.array:
    """ISUP rule, Gleason scores"""
    scores = np.zeros((px_per_grades.shape[0], px_per_grades.shape[1])).astype('int')
    for idi in range(px_per_grades.shape[0]):
        for idm in range(px_per_grades.shape[1]):
            if px_per_grades[idi, idm].sum() > 0:
                h = px_per_grades[idi, idm][2:]
                if h.sum() == 0:
                    scores[idi, idm] = 0  # all benign
                else:
                    s = np.argsort(h)[::-1]

                    p1 = s[0]
                    p2 = s[1]
                    # if there's no second pattern: double the first:
                    if h[p2] == 0:
                        p2 = p1
                    # if grade of second pattern is lower: check the 5% rule
                    elif p2 < p1:
                        if h[p2] < 0.05 * h.sum():
                            p2 = p1

                    scores[idi, idm] = p2 + 2 + p1 + 2
    return scores


def get_E_ISUP(px_per_grades: np.array) -> np.array:
    """ISUP rule, Epstein groups"""
    scores = np.ones((px_per_grades.shape[0], px_per_grades.shape[1])).astype('int')
    for idi in range(px_per_grades.shape[0]):
        for idm in range(px_per_grades.shape[1]):
            if px_per_grades[idi, idm].sum() > 0:
                h = px_per_grades[idi, idm][2:]
                if h.sum() == 0:
                    scores[idi, idm] = 1
                else:
                    s = np.argsort(h)[::-1]

                    p1 = s[0]
                    p2 = s[1]

                    # if there's no second pattern: double the first:
                    if h[p2] == 0:
                        p2 = p1
                    # if grade of second pattern is lower: check the 5% rule
                    elif p2 < p1:
                        if h[p2] < 0.05 * h.sum():
                            p2 = p1

                    scores[idi, idm] = epstein(p1 + 2, p2 + 2, h[p1], h[p2])
    return scores


def get_kappas(scores: np.array, has_annotations: np.array, labels: List, w: str = None) -> Tuple[np.array, np.array]:
    """Computing the 1 v 1 kappas between a set of annotators.
    Also returns the number of overlapping samples for each pair."""
    kappas = np.zeros((has_annotations.shape[1], has_annotations.shape[1]))
    kappas_n = np.zeros((has_annotations.shape[1], has_annotations.shape[1]))
    for i in range(has_annotations.shape[1]):
        for j in range(i + 1, has_annotations.shape[1]):
            mask = (has_annotations[:, i] * has_annotations[:, j]) > 0
            kappas_n[i, j] = mask.sum()
            kappas[i, j] = cohen_kappa_score(scores[mask, i], scores[mask, j], weights=w, labels=labels)

    kappas = kappas + kappas.T
    kappas[np.eye(kappas.shape[0]) == 1] = 1
    kappas_n += kappas_n.T

    return kappas, kappas_n


def compute_scores(px_per_grades: np.array) -> Tuple:
    """Compute the results of the different scoring functions on the px_per_grades"""

    return get_G_SP(px_per_grades), get_E_SP(px_per_grades), get_G_ISUP(px_per_grades), get_E_ISUP(px_per_grades)

