"""
Version to replicate the results from SNOW preprint on GlaS, but also compute the MCC on it.
"""
import math

import tensorflow as tf
import os
import numpy as np
from WarwickDataFeed import WarwickDataFeed


def softmax2(k):
    e = np.exp(k - k.max())
    s = np.exp(k - k.max()).sum(axis=-1)
    s[s == 0] = 1
    return e[..., 0] / s


def Dice(T, P):
    TP = ((T == 1) * (P == 1)).sum()
    FP = ((T == 0) * (P == 1)).sum()
    return TP / (T.sum() + FP)


def F1(T, P):
    TP = ((T == 1) * (P == 1)).sum()
    TN = ((T == 0) * (P == 0)).sum()
    FP = ((T == 0) * (P == 1)).sum()
    FN = ((T == 1) * (P == 0)).sum()

    recall = TP / (TP + FN) if TP + FN > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    if (recall + precision) == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def MCC(T, P):
    TP = float(((T == 1) * (P == 1)).sum())
    TN = float(((T == 0) * (P == 0)).sum())
    FP = float(((T == 0) * (P == 1)).sum())
    FN = float(((T == 1) * (P == 0)).sum())

    mcc = (TP * TN - FP * FN) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    if TP+FP == 0:
        mcc = 0

    return mcc


def main():
    """clfs_no_offset = ['F-Norm-4', 'F-R50', 'F-BB', 'F-R50-BB', 'F-R50-HD',
                      'GEN-anno', 'GEN-R50', 'GEN-BB', 'GEN-R50-BB', 'GEN-R50-HD',
                      'SS0-anno', 'SS0-R50', 'SS0-BB', 'SS0-R50-BB', 'SS0-R50-HD',
                      'SSP-anno', 'SSP-r50', 'SSP-BB', 'SSP-R50-BB', 'SSP-R50-HD']
    clfs_offset = ['GSN-anno-datacentered', 'GSN-r50-datacentered', 'GSN-bb-datacentered', 'GSN-r50-bb-datacentered',
                   'GSN-r50-hd-datacentered',
                   'GSN75-anno-datacentered', 'GSN75-r50-datacentered', 'GSN75-bb-datacentered',
                   'GSN75-r50-bb-datacentered', 'GSN75-r50-hd-datacentered',
                   'tda-anno', 'tda-r50', 'tda-bb', 'tda-r50-bb', 'tda-r50-hd']"""
    # Check nans:
    clfs_no_offset = ['F-R50-HD', 'SS0-R50-HD']
    clfs_offset = []

    clfs = clfs_no_offset + clfs_offset

    for clf_name in clfs:
        X_OFFSET = 0 if clf_name in clfs_no_offset else 0.5
        print(clf_name)

        params = {
            "clf_name": clf_name,
            "tile_size": 256,
            "summaries_dir": "e:/data/tf_summaries/warwick_snow",
            "checkpoints_dir": "e:/data/tf_checkpoint/warwick_snow",
            "dataset_dir": "e:/data/GlaS",
            "annotations": "anno"
        }
        CHECKPOINTS_DIR = params['checkpoints_dir']
        overlap = 2
        tile_size = params['tile_size']

        tf.reset_default_graph()
        sess = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(CHECKPOINTS_DIR, '%s_best.ckpt.meta' % clf_name))
        saver.restore(sess, os.path.join(CHECKPOINTS_DIR, '%s_best.ckpt' % clf_name))

        try:
            X = tf.get_default_graph().get_tensor_by_name("features/X:0")
        except KeyError:
            X = tf.get_default_graph().get_tensor_by_name("ae/features/X:0")

        try:
            tensorname = "output/segmentation:0"
            Y_seg = tf.get_default_graph().get_tensor_by_name(tensorname)
        except KeyError:
            tensorname = 'classifier/segmentation/LeakyRelu/Maximum:0'
            lrelu = tf.get_default_graph().get_tensor_by_name(tensorname)
            Y_seg = tf.nn.softmax(lrelu, name='output/segmentation')

        f1s = []
        mccs = []

        for ds in ['train', 'testA', 'testB']:

            data = WarwickDataFeed(params, ds)

            for idx in range(len(data.files_X)):
                im = data.images_X[idx] / 255. - X_OFFSET  # imread(data.files_X[idx])
                im_anno = data.images_Y[idx] > 0  # imread(data.files_Y[idx]) > 0

                imshape = im_anno.shape
                nr, nc = (
                    overlap * np.ceil((imshape[0] - 1) / tile_size), overlap * np.ceil((imshape[1] - 1) / tile_size))
                yr, xr = (np.arange(0, nr) * ((imshape[0] - 1 - tile_size) // (nr - 1))).astype('int'), (
                        np.arange(0, nc) * ((imshape[1] - 1 - tile_size) // (nc - 1))).astype('int')
                mesh = np.meshgrid(yr, xr)
                tiles = zip(mesh[0].flatten(), mesh[1].flatten())

                im_pred = np.zeros_like(im_anno).astype('float')
                for t in tiles:
                    batch_X = [im[t[0]:t[0] + tile_size,
                               t[1]:t[1] + tile_size]]  # [im[t[0]:t[0]+tile_size, t[1]:t[1]+tile_size]/255.]
                    sm = Y_seg.eval(session=sess, feed_dict={X: batch_X})[:, :, :, 0]
                    im_pred[t[0]:t[0] + tile_size, t[1]:t[1] + tile_size] = np.maximum(
                        im_pred[t[0]:t[0] + tile_size, t[1]:t[1] + tile_size], sm[0, :, :])

                mask_pred = im_pred > 0.5

                f1 = F1(im_anno, mask_pred)
                mcc = MCC(im_anno, mask_pred)

                f1s += [("%f\n" % f1).replace('.', ',')]
                mccs += [("%f\n" % mcc).replace('.', ',')]

        with open('results_warwick/f1-check/res-f1-%s.txt' % clf_name, 'w') as fp:
            fp.writelines(f1s)
        with open('results_warwick/f1-check/res-mcc-%s.txt' % clf_name, 'w') as fp:
            fp.writelines(mccs)


if __name__ == "__main__":
    main()
