import csv
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np

MITOS12_PATH = "e:/data/MITOS12"
PX_SIZE = 0.2456  # µm/px
MAX_DISTANCE_mu = 5  # µm
MAX_DISTANCE_px = MAX_DISTANCE_mu/PX_SIZE  # px
PATCH_SIZE = (2048, 2048)


def _get_path(train: bool = True) -> str:
    if train:
        return os.path.join(MITOS12_PATH, 'train')
    else:
        return os.path.join(MITOS12_PATH, 'test')


def get_file_names(train: bool = True) -> List[str]:
    path = _get_path(train)

    return [f.split('.')[0] for f in os.listdir(path) if f.split('.')[1] == 'csv']


@dataclass
class BoundingBox:
    x0: int
    x1: int
    y0: int
    y1: int

    @property
    def shape(self) -> Tuple[int, int]:
        return abs(self.x1-self.x0), abs(self.y1-self.y0)

    @property
    def area(self) -> int:
        return self.shape[0]*self.shape[1]


@dataclass
class Mitosis:
    bbox: BoundingBox
    centroid: Tuple[float, float]
    area: int
    px: np.array


def get_mitosis_per_file(files: List[str], train: bool = True) -> Dict[str, List[Mitosis]]:
    path = _get_path(train)

    mitosis_per_file = {}
    for f in files:
        mitosis_per_file[f] = []
        with open(os.path.join(path, f'{f}.csv'), 'r') as fp:
            reader = csv.reader(fp)
            for row in reader:
                mitosis = np.array([int(r) for r in row])
                xs = mitosis[::2]
                ys = mitosis[1::2]
                bbox = BoundingBox(x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max())
                mitosis_per_file[f].append(Mitosis(bbox=bbox,
                                                   centroid=(xs.mean(), ys.mean()),
                                                   area=len(xs),
                                                   px=np.array([xs, ys])))

    return mitosis_per_file


def get_total_candidate_area(files: List[str]) -> int:
    px_per_file = PATCH_SIZE[0]*PATCH_SIZE[1]
    return px_per_file*len(files)


def detector_simulator(positive_examples: int,
                       negative_candidates: int,
                       sensitivity: float = 0.9,
                       specificity: float = 0.99,
                       pre_detector_specificity: float = 0.99) -> np.array:
    tp = positive_examples*sensitivity
    fn = positive_examples*(1-sensitivity)
    fp = negative_candidates*(1-pre_detector_specificity)*(1-specificity)
    tn = negative_candidates*pre_detector_specificity + negative_candidates*(1-pre_detector_specificity)*specificity
    return tp, fp, fn, tn

