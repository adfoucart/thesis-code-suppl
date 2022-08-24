from typing import List, Dict, Callable, Tuple
import numpy as np
from matplotlib import pyplot as plt
from metrics.segmentation import IoU_cm, get_fuzzy_weights, get_removed_weights
from datasets.monusac import CELL_TYPES, get_all_nuclei_regions

"""
@author Adrien Foucart

Experiments on the effect of small errors on the IoU in small objects.

Require the MoNuSAC 2020 testing annotations, available from : https://monusac-2020.grand-challenge.org/Data/
Set the MONUSAC_PATH variable below to the "MoNuSAC Testing Data and Annotations" folder in the datasets/monusac.py file.
"""


def plot_class_hists(values: Dict[str, List[float]], save_to: str = None) -> None:
    for cl, cl_values in values.items():
        plt.figure()
        plt.hist(cl_values, bins=100)
        plt.title(cl)
        if save_to is not None:
            plt.savefig(f'{save_to}_{cl}.png')
        else:
            plt.show()


def plot_class_boxplots(values: Dict[str, List[float]], y_label: str = "", save_to: str = None) -> None:
    for cl, cl_values in values.items():
        values[cl] = [v for v in cl_values if v is not None]
    plt.figure()
    plt.boxplot(values.values(), labels=values.keys())
    plt.ylabel(y_label)
    if save_to is not None:
        plt.savefig(f'{save_to}.png')
    else:
        plt.show()


def plot_class_scatterplots(values_x: Dict[str, List[float]],
                            values_y: Dict[str, List[float]],
                            label_x: str,
                            label_y: str,
                            figsize: Tuple[int] = (10, 8),
                            save_to: str = None) -> None:
    plt.figure(figsize=figsize)
    for cl in CELL_TYPES:
        xval = []
        yval = []
        for x,y in zip(values_x[cl], values_y[cl]):
            if x is None or y is None:
                continue
            xval.append(x)
            yval.append(y)
        plt.scatter(xval, yval, label=cl, marker='+')
    plt.legend()
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    if save_to is not None:
        plt.savefig(f'{save_to}.png')
    else:
        plt.show()


def plot_IoU_incremental_shift(bbox: np.array,
                               n_shifts: int = 5,
                               weight_func: Callable = None,
                               n: int = 1,
                               save_to: str = None) -> None:
    bbox_ref = np.zeros_like(bbox)
    plt.figure(figsize=(20,5))
    for shift in range(0, n_shifts):
        bbox_ref[shift:] = bbox[:bbox.shape[0] - shift]
        iou = IoU_cm(bbox > 0, bbox_ref > 0, weight_func=weight_func, n=n)
        plt.subplot(1, n_shifts, shift + 1)
        plt.imshow(bbox_ref)
        plt.contour(bbox)
        plt.title(f'IoU = {iou:.2f}')
    if save_to is not None:
        plt.savefig(f'{save_to}.png')
    else:
        plt.show()


def get_all_IoUs_shift(bboxes: Dict[str, List[np.array]],
                       shift: int = 1,
                       weight_func: Callable = None,
                       n: int = 1,) -> Dict[str, List[float]]:
    IoUs_px_shift = {}
    for cl, bbox_class in bboxes.items():
        IoUs_px_shift[cl] = []
        for bbox in bbox_class:
            bbox_ref = np.zeros_like(bbox)
            bbox_ref[shift:] = bbox[:bbox.shape[0] - shift]
            IoUs_px_shift[cl].append(IoU_cm(bbox > 0, bbox_ref > 0, weight_func=weight_func, n=n))
    return IoUs_px_shift


def main():
    areas, bboxes = get_all_nuclei_regions()
    plot_class_hists(areas, save_to='areas_per_class_hist')
    plot_class_boxplots(areas, y_label='Area', save_to='areas_per_class_boxplots')

    # -- Regular CM --
    IoUs_single_px_shift = get_all_IoUs_shift(bboxes, 1)
    plot_class_boxplots(IoUs_single_px_shift, y_label='IoU (single px shift)', save_to='IoU_per_class_boxplots_normal')
    plot_class_scatterplots(areas, IoUs_single_px_shift, 'Area (px)', 'IoU', save_to='IoU_vs_area_singlepx_normal')
    plot_IoU_incremental_shift(bboxes[CELL_TYPES[1]][0], n_shifts=5, save_to='IoU_shift_normal')

    # -- Fuzzy CM / Ignore borders version --
    IoUs_single_px_shift = {}
    for cl in CELL_TYPES:
        IoUs_single_px_shift[cl] = []
        for bbox in bboxes[cl]:
            bbox_ref = np.zeros_like(bbox)
            bbox_ref[1:] = bbox[:bbox.shape[0] - 1]
            IoUs_single_px_shift[cl].append(IoU_cm(bbox > 0, bbox_ref > 0, weight_func=get_removed_weights, n=4))

    plot_class_boxplots(IoUs_single_px_shift, y_label='IoU (single px shift)', save_to='IoU_per_class_boxplots_ignore')
    plot_class_scatterplots(areas, IoUs_single_px_shift, 'Area (px)', 'IoU', save_to='IoU_vs_area_singlepx_ignore')
    plot_IoU_incremental_shift(bboxes[CELL_TYPES[1]][0], n_shifts=5, weight_func=get_removed_weights, n=4, save_to='IoU_shift_ignore')

    # -- Fuzzy CM / Weighted border region version --
    IoUs_single_px_shift = {}
    for cl in CELL_TYPES:
        IoUs_single_px_shift[cl] = []
        for bbox in bboxes[cl]:
            bbox_ref = np.zeros_like(bbox)
            bbox_ref[1:] = bbox[:bbox.shape[0] - 1]
            IoUs_single_px_shift[cl].append(IoU_cm(bbox > 0, bbox_ref > 0, weight_func=get_fuzzy_weights, n=4))

    plot_class_boxplots(IoUs_single_px_shift, y_label='IoU (single px shift)', save_to='IoU_per_class_boxplots_weighted')
    plot_class_scatterplots(areas, IoUs_single_px_shift, 'Area (px)', 'IoU', save_to='IoU_vs_area_singlepx_weighted')
    plot_IoU_incremental_shift(bboxes[CELL_TYPES[1]][0], n_shifts=5, weight_func=get_fuzzy_weights, n=4, save_to='IoU_shift_weighted')


if __name__ == "__main__":
    main()
