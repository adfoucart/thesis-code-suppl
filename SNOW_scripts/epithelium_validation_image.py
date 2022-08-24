# Epithelium validation
# Using Per-Pixel F1 & ROC AUC

import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from EpitheliumDataFeed import EpitheliumDataFeed
from skimage.io import imread, imsave
from dhutil.tools import F1, MCC

clfs = ['shortres-epi-baseline',
        'shortres-epi-defo',
        'shortres-epi-n50',
        'shortres-epi-full-onlyP',
        'shortres-epi-n50-onlyP',
        'shortres-epi-defo-onlyP',
        'shortres-epi-full-GSN100',
        'shortres-epi-n50-GSN100',
        'shortres-epi-defo-GSN100',
        'shortres-epi-full-tda',
        'shortres-epi-n50-tda',
        'shortres-epi-defo-tda',
        'ShortRes-SSL-OnlyP-full',
        'ShortRes-SSL-OnlyP-noisy',
        'ShortRes-SSL-OnlyP-defo']

X_OFFSET = 0.5  # Use 0 for older networks, 0.5 for networks trained on 0-centered data.

WITH_MORPHOLOGY = False
resdirapp = '-pp' if WITH_MORPHOLOGY else ''

params = {
    "clf_name": "",
    "tile_size": 128,
    "summaries_dir": "e:/data/tf_summaries/epi",
    "checkpoints_dir": "e:/data/tf_checkpoint/epi",
    "dataset_dir": "E:/data/Epithelium"
}

ds = 'test'
data = EpitheliumDataFeed(params, ds)
overlap = 2

for idx in [1, 5, 6]:
    im = data.images_X[idx] / 255. - X_OFFSET  # imread(data.files_X[idx])
    im_anno = data.images_Y[idx] > 0  # imread(data.files_Y[idx]) > 0

    im_out = im + 0.5
    overlay = np.zeros((im_out.shape[0], im_out.shape[1], 4)).astype('float')
    overlay[im_anno, 0] = 1.
    overlay[im_anno, 3] = 0.5

    plt.figure(figsize=(20, 20))
    plt.imshow(im_out)
    plt.imshow(overlay)
    plt.axis('off')
    plt.savefig("../Results/epithelium_2022/images/anno-%s-%d.png" % (ds, idx))

    for clf_name in clfs:
        print(clf_name)

        tile_size = 256 if clf_name.find('-256') >= 0 else 128

        imshape = im_anno.shape
        ny = 1 + overlap * (imshape[0] // tile_size)
        nx = 1 + overlap * (imshape[1] // tile_size)
        yts = [i * (imshape[0] - tile_size) // ny for i in range(ny + 1)]
        xts = [i * (imshape[1] - tile_size) // nx for i in range(nx + 1)]
        mesh = np.meshgrid(yts, xts)
        tiles = zip(mesh[0].flatten(), mesh[1].flatten())

        params["clf_name"] = clf_name
        params["tile_size"] = tile_size

        CHECKPOINTS_DIR = params['checkpoints_dir'] if 'SSL-OnlyP' not in clf_name else f"{params['checkpoints_dir']}_2022"

        tf.reset_default_graph()
        sess = tf.Session()
        saver = tf.train.import_meta_graph(os.path.join(CHECKPOINTS_DIR, '%s_best.ckpt.meta' % clf_name))
        saver.restore(sess, os.path.join(CHECKPOINTS_DIR, '%s_best.ckpt' % clf_name))

        try:
            X = tf.get_default_graph().get_tensor_by_name("features/X:0")
        except KeyError:
            X = tf.get_default_graph().get_tensor_by_name("ae/features/X:0")

        tensorname = "output/segmentation:0"
        Y_seg = tf.get_default_graph().get_tensor_by_name(tensorname)

        im_pred = np.zeros_like(im_anno).astype('float')
        for t in tiles:
            batch_X = [im[t[0]:t[0] + tile_size, t[1]:t[1] + tile_size]]
            sm = Y_seg.eval(session=sess, feed_dict={X: batch_X})[:, :, :, 0]
            im_pred[t[0]:t[0] + tile_size, t[1]:t[1] + tile_size] = np.maximum(
                im_pred[t[0]:t[0] + tile_size, t[1]:t[1] + tile_size], sm[0, :, :])

        mask_pred = im_pred > 0.5

        im_out = im+0.5
        overlay = np.zeros((im_out.shape[0], im_out.shape[1], 4)).astype('float')
        overlay[mask_pred, 0] = 1.
        overlay[mask_pred, 3] = 0.5

        plt.figure(figsize=(20, 20))
        plt.imshow(im_out)
        plt.imshow(overlay)
        plt.axis('off')
        plt.savefig("../Results/epithelium_2022/images/res-%s-%d-%s.png" % (ds, idx, clf_name))


    """TP = (mask_pred == 1) * (im_anno == 1)
    TN = (mask_pred == 0) * (im_anno == 0)
    FP = (mask_pred == 1) * (im_anno == 0)
    FN = (mask_pred == 0) * (im_anno == 1)

    im_out = np.zeros_like(data.images_X[idx])
    im_out[TP, :] = np.array([255, 255, 255])
    im_out[FP, :] = np.array([0, 0, 255])
    im_out[FN, :] = np.array([255, 0, 0])

    # imsave("E:/pCloud/ULB/Doctorat/Publications/2020XX-SNOW2/epi/res-%s-%d-%s.png"%(ds,idx,clf_name), im_out)
    # imsave("E:/pCloud/ULB/Doctorat/Rapports/SNOW/pub-MedImAnal/epi/res-%s-%d-%s.png"%(ds,idx,clf_name), im_out)
    imsave("../Results/epithelium_2022/images/res-%s-%d-%s.png" % (ds, idx, clf_name), im_out)"""

# im_out = data.images_X[idx].copy()
# overlay = np.zeros((im_out.shape[0],im_out.shape[1],4)).astype('float')
# overlay[mask_pred,0] = 1.
# overlay[mask_pred,3] = 0.5

# plt.figure()
# plt.imshow(im_out)
# plt.imshow(overlay)
# plt.axis('off')
# plt.savefig("E:/pCloud/ULB/Doctorat/Publications/2020XX-SNOW2/epi/%s-%d-%s.png"%(ds,idx,clf_name))
