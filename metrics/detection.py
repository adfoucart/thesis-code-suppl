from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np


def precision(tp: int, fp: int, fn: int) -> float:
    if tp + fp == 0:
        return 1
    return tp/(tp+fp)


def recall(tp: int, fp: int, fn: int) -> Optional[float]:
    if tp + fn == 0:
        return None
    return tp/(tp+fn)


def f1(tp: int, fp: int, fn: int) -> Optional[float]:
    prec = precision(tp, fp, fn)
    rec = recall(tp, fp, fn)
    if rec is None:
        return None
    if prec + rec == 0:
        return 0
    return 2*prec*rec/(prec+rec)


def _ap(pr_points, sampling=100):
    r_samples = np.arange(0, 1+1/sampling, 1/sampling)
    cur_point = 0
    cur_auc = 0
    for r in r_samples:
        prev_point = cur_point
        for p in pr_points[cur_point:]:
            if p[1] >= r:
                cur_auc += pr_points[cur_point:, 0].max()/(sampling+1)
                cur_point = prev_point+1
                break
            prev_point += 1
    return cur_auc


def average_precision(gt: np.array, pred_confidence: np.array) -> Tuple[float, np.array]:
    """Return AP & PR-Curve"""
    if len(gt) != len(pred_confidence):
        raise ValueError("gt and pred_confidence should have the same size")

    tp = 0
    fn = (gt == 1).sum()
    fp = 0
    tn = (gt == 0).sum()
    conf_sorted = np.argsort(pred_confidence)[::-1]

    cm = tp, fp, fn
    pr_points = [(precision(*cm), recall(*cm))]

    for idx in conf_sorted:
        if gt[idx]:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1

        cm = tp, fp, fn
        pr_points.append((precision(*cm), recall(*cm)))
    pr_points = np.array(pr_points)
    return _ap(pr_points), pr_points


@dataclass
class Match:
    gt_idx: int
    pred_idx: int
    distance: float
    confidence: float


def get_cm(matches: List[Match],
           gt_idxs: List[int],
           pred_idxs: List[int],
           p_confidence: List[float],
           thresh_conf: float,
           thresh_dist: float):
    TP = FP = FN = 0
    gt_remaining = [gt for gt in gt_idxs]
    pred_remaining = [pred for pred in pred_idxs]
    for match in matches:
        if match.confidence >= thresh_conf and match.distance <= thresh_dist:
            TP += 1
            gt_remaining.remove(match.gt_idx)
            pred_remaining.remove(match.pred_idx)
    for pred in pred_remaining:
        if p_confidence[pred] >= thresh_conf:
            FP += 1
    FN = len(gt_remaining)
    return TP, FP, FN
