# Epithelium validation
# Using Per-Pixel F1 & ROC AUC
from typing import Optional, Tuple

import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from EpitheliumDataFeed import EpitheliumDataFeed
from skimage.io import imread
from skimage.morphology import opening, closing, disk
from sklearn.metrics import auc
from dhutil.tools import F1, MCC

clfs = ['ShortRes-SSL-OnlyP-full',
        'shortres-epi-baseline',
        'shortres-epi-full-onlyP',
        'shortres-epi-full-GSN100',
        'shortres-epi-full-tda',
        'shortres-epi-n50',
        'ShortRes-SSL-OnlyP-noisy',
        'shortres-epi-n50-onlyP',
        'shortres-epi-n50-GSN100',
        'shortres-epi-n50-tda',
        'shortres-epi-defo',
        'ShortRes-SSL-OnlyP-defo',
        'shortres-epi-defo-onlyP',
        'shortres-epi-defo-GSN100',
        'shortres-epi-defo-tda',
        'PAN-SSL-OnlyP-defo',
        'PAN-SSL-OnlyP-noisy',
        'PAN-SSL-OnlyP-full',
        'pan-epi-full-baseline',
        'pan-epi-n50-baseline',
        'pan-epi-defo-baseline',
        'pan-epi-full-onlyP',
        'pan-epi-n50-onlyP',
        'pan-epi-defo-onlyP',
        'pan-epi-defo-tda',
        'pan-epi-defo-GSN100',
        'pan-epi-full-tda',
        'pan-epi-full-GSN100',
        'pan-epi-n50-tda',
        'pan-epi-n50-GSN100']
# clfs = ['unet-epi-full-baseline']

# clfs = ['shortres-epi-full-GSN100', 'shortres-epi-defo-GSN100', 'shortres-epi-n50-GSN100', 'shortres-epi-n50-bb-GSN100']
# clfs = ['shortres-epi-defo', 'shortres-epi-full-GSN100']
# clfs = ['pan-epi-full-GSN100', 'pan-epi-defo-GSN100', 'pan-epi-n50-GSN100', 'pan-epi-n50-bb-GSN100']
X_OFFSET = 0.5  # Use 0 for older networks, 0.5 for networks trained on 0-centered data.

WITH_MORPHOLOGY = False
resdirapp = '-pp' if WITH_MORPHOLOGY else ''

for clf_name in clfs:
    print(clf_name)

    tile_size = 256 if clf_name.find('-256') >= 0 else 128

    params = {
        "clf_name": clf_name,
        "tile_size": tile_size,
        "summaries_dir": "e:/data/tf_summaries/epi",
        "checkpoints_dir": "e:/data/tf_checkpoint/epi",
        "dataset_dir": "E:/data/Epithelium"
    }
    CHECKPOINTS_DIR = params['checkpoints_dir'] if 'SSL-OnlyP' not in clf_name else f"{params['checkpoints_dir']}_2022"

    overlap = 2

    tf.reset_default_graph()
    sess = tf.Session()
    saver = tf.train.import_meta_graph(os.path.join(CHECKPOINTS_DIR, '%s_best.ckpt.meta' % (clf_name)))
    saver.restore(sess, os.path.join(CHECKPOINTS_DIR, '%s_best.ckpt' % clf_name))

    try:
        X = tf.get_default_graph().get_tensor_by_name("features/X:0")
    except KeyError:
        X = tf.get_default_graph().get_tensor_by_name("ae/features/X:0")

    tensorname = "output/segmentation:0"
    Y_seg = tf.get_default_graph().get_tensor_by_name(tensorname)

    f1s = []
    mccs = []

    #for ds in ['train', 'test']:
    for ds in ['test']:
        data = EpitheliumDataFeed(params, ds)

        for idx in range(len(data.files_X)):
            im = data.images_X[idx] / 255. - X_OFFSET  # imread(data.files_X[idx])
            im_anno = data.images_Y[idx] > 0  # imread(data.files_Y[idx]) > 0

            imshape = im_anno.shape
            ny = 1 + overlap * (imshape[0] // tile_size)
            nx = 1 + overlap * (imshape[1] // tile_size)
            yts = [i * (imshape[0] - tile_size) // ny for i in range(ny + 1)]
            xts = [i * (imshape[1] - tile_size) // nx for i in range(nx + 1)]
            mesh = np.meshgrid(yts, xts)
            tiles = zip(mesh[0].flatten(), mesh[1].flatten())

            im_pred = np.zeros_like(im_anno).astype('float')
            for t in tiles:
                batch_X = [im[t[0]:t[0] + tile_size, t[1]:t[1] + tile_size]]
                sm = Y_seg.eval(session=sess, feed_dict={X: batch_X})[:, :, :, 0]
                im_pred[t[0]:t[0] + tile_size, t[1]:t[1] + tile_size] = np.maximum(
                    im_pred[t[0]:t[0] + tile_size, t[1]:t[1] + tile_size], sm[0, :, :])

            np.save(f'e:/pCloud/ULB/Doctorat/workspace/Results/epithelium_2022/predictions/{clf_name}_{ds}_{idx}.npy', im_pred)
            """mask_pred = im_pred > 0.5

            f1 = F1(im_anno, mask_pred)
            mcc = MCC(im_anno, mask_pred)

            f1s += [("%f\n" % f1).replace('.', ',')]
            mccs += [("%f\n" % mcc).replace('.', ',')]

    with open('results_epi/%s-f1.txt' % clf_name, 'w') as fp:
        fp.writelines(f1s)
    with open('results_epi/%s-mcc.txt' % clf_name, 'w') as fp:
        fp.writelines(mccs)"""
