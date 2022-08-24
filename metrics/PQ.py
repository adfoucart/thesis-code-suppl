import numpy as np


def Panoptic_quality(ground_truth_image, predicted_image, return_stats=False):
    """Panoptic Quality computation from Verma et al.
    https://github.com/ruchikaverma-iitg/MoNuSAC/blob/3039c9011ed3f88d2219834fcd332bf2207e79b2/PQ_metric.ipynb"""
    TP = 0
    FP = 0
    FN = 0
    sum_IOU = 0
    matched_instances = {}  # Create a dictionary to save ground truth indices in keys and predicted matched instances as velues
    # It will also save IOU of the matched instance in [indx][1]

    # Find matched instances and save it in a dictionary
    for i in np.unique(ground_truth_image):
        if i == 0:
            pass
        else:
            temp_image = np.array(ground_truth_image)
            temp_image = temp_image == i
            matched_image = temp_image * predicted_image

            for j in np.unique(matched_image):
                if j == 0:
                    pass
                else:
                    pred_temp = predicted_image == j
                    intersection = sum(sum(temp_image * pred_temp))
                    union = sum(sum(temp_image + pred_temp))
                    IOU = intersection / union
                    if IOU > 0.5:
                        matched_instances[i] = j, IOU

                        # Compute TP, FP, FN and sum of IOU of the matched instances to compute Panoptic Quality

    pred_indx_list = np.unique(predicted_image)
    pred_indx_list = np.array(pred_indx_list[1:])

    # Loop on ground truth instances
    for indx in np.unique(ground_truth_image):
        if indx == 0:
            pass
        else:
            if indx in matched_instances.keys():
                pred_indx_list = np.delete(pred_indx_list, np.argwhere(pred_indx_list == [indx][0]))
                TP = TP + 1
                sum_IOU = sum_IOU + matched_instances[indx][1]
            else:
                FN = FN + 1
    FP = len(np.unique(pred_indx_list))
    PQ = sum_IOU / (TP + 0.5 * FP + 0.5 * FN)

    if return_stats:
        return PQ, TP, FP, FN, sum_IOU/max(1, TP)

    return PQ


def Panoptic_quality_corrected(ground_truth_image, predicted_image, return_stats = False):
    TP = 0
    FP = 0
    FN = 0
    sum_IOU = 0
    matched_instances = {}  # Create a dictionary to save ground truth indices in keys and predicted matched instances as velues
    # It will also save IOU of the matched instance in [indx][1]

    # Find matched instances and save it in a dictionary
    for i in np.unique(ground_truth_image):
        if i == 0:
            pass
        else:
            temp_image = np.array(ground_truth_image)
            temp_image = temp_image == i
            matched_image = temp_image * predicted_image

            for j in np.unique(matched_image):
                if j == 0:
                    pass
                else:
                    pred_temp = predicted_image == j
                    intersection = sum(sum(temp_image * pred_temp))
                    union = sum(sum(temp_image + pred_temp))
                    IOU = intersection / union
                    if IOU > 0.5:
                        matched_instances[i] = j, IOU

                        # Compute TP, FP, FN and sum of IOU of the matched instances to compute Panoptic Quality

    pred_indx_list = np.unique(predicted_image)
    pred_indx_list = np.array(pred_indx_list[1:])

    # Loop on ground truth instances
    for indx in np.unique(ground_truth_image):
        if indx == 0:
            pass
        else:
            if indx in matched_instances.keys():
                pred_indx_list = np.delete(pred_indx_list, np.argwhere(pred_indx_list == matched_instances[indx][0]))
                TP = TP + 1
                sum_IOU = sum_IOU + matched_instances[indx][1]
            else:
                FN = FN + 1
    FP = len(np.unique(pred_indx_list))
    PQ = sum_IOU / (TP + 0.5 * FP + 0.5 * FN)

    if return_stats:
        return PQ, TP, FP, FN, sum_IOU/max(1, TP)

    return PQ
