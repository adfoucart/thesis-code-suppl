import os

import numpy as np
from skimage.io import imread
from skimage.transform import downscale_local_mean, resize
from skimage.color import rgb2hsv
from skimage.morphology import opening, closing, disk


def get_tissue_mask(rgb: np.array, scale_factor: int=8) -> np.array:
    """Return image mask with
    0 = background (no tissue),
    1 = foreground (tissue).

    Background mask is computed at a lower resolution with scale_factor
    """

    lr = downscale_local_mean(rgb, (scale_factor, scale_factor, 1))
    hsv = rgb2hsv(lr)

    bg = hsv[:, :, 1] < 0.04
    bg = resize(bg, (rgb.shape[0], rgb.shape[1])) < 0.5
    bg = opening(closing(bg, disk(5)), disk(10)).astype('bool')

    return bg


def get_visualisation_supervised(gt_mask: np.array, pred_mask: np.array, tissue_mask: np.array) -> np.array:
    TP = pred_mask * gt_mask * tissue_mask
    FP = pred_mask * (gt_mask == False) * tissue_mask
    FN = (pred_mask == False) * gt_mask * tissue_mask
    TN = (pred_mask == False) * (gt_mask == False) * tissue_mask

    results_viz = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype='uint8')
    results_viz[TP] = np.array([0, 255, 0])
    results_viz[FP] = np.array([255, 0, 0])
    results_viz[FN] = np.array([0, 0, 255])
    results_viz[TN] = np.array([255, 255, 255])

    return results_viz


def load_predictions_SNOW(directory: str):
    prob_maps = {}
    networks = ['pan', 'shortres', 'unet']
    marks = ['s', 'o', 'd']
    strategies = ['baseline', 'gsn50', 'onlyP']
    colors = ['k', 'r', 'b']

    for network in networks:
        prob_maps[network] = {}
        for strat in strategies:
            prob_maps[network][strat] = []
            tiles_res_dir = os.path.join(directory, f"{network}-artefact-{strat}")
            for i in range(21):
                p = imread(os.path.join(tiles_res_dir, f"{i:02}_rgb.png_proba.png"))
                if network == 'unet':
                    m = p / 255
                else:
                    m = p / 65535
                prob_maps[network][strat].append(m.flatten())
    return prob_maps
