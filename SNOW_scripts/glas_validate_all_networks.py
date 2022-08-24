import tensorflow as tf
import numpy as np
from data import WarwickDataFeed
from dhutil.tools import F1, MCC, ACC, JACC, printProgressBar
import json

clfs = {
    'F-Norm-4': 0.,
    'F-R50': 0.,
    'F-BB': 0.,
    'F-R50-BB': 0.,
    'F-R50-HD': 0.,
    'GEN-anno': 0.,
    'GEN-R50': 0.,
    'GEN-BB': 0.,
    'GEN-R50-BB': 0.,
    'GEN-R50-HD': 0.,
    'SS0-anno': 0.,
    'SS0-R50': 0.,
    'SS0-BB': 0.,
    'SS0-R50-BB': 0.,
    'SS0-R50-HD': 0.,
    'SSP-anno': 0.,
    'SSP-r50': 0.,
    'SSP-BB': 0.,
    'SSP-R50-BB': 0.,
    'SSP-R50-HD': 0.,
    'GSN-anno-datacentered': 0.5,
    'GSN-r50-datacentered': 0.5,
    'GSN-bb-datacentered': 0.5,
    'GSN-r50-bb-datacentered': 0.5,
    'GSN-r50-hd-datacentered': 0.5,
    'GSN75-anno-datacentered': 0.5,
    'GSN75-r50-datacentered': 0.5,
    'GSN75-bb-datacentered': 0.5,
    'GSN75-r50-bb-datacentered': 0.5,
    # 'GSN75-r50-hd-datacentered': 0.5,
    'tda-anno': 0.5,
    'tda-r50': 0.5,
    'tda-bb': 0.5,
    'tda-r50-bb': 0.5,
    'tda-r50-hd': 0.5
}

params = {
    "clf_name": "",
    "tile_size": 256,
    "summaries_dir": "e:/data/tf_summaries/warwick_snow",
    "checkpoints_dir": "e:/data/tf_checkpoint/warwick_snow",
    "dataset_dir": "e:/data/GlaS",
    "annotations": "anno"
}
CHECKPOINTS_DIR = params['checkpoints_dir']
overlap = 2
tile_size = params['tile_size']

# Pre-loading all data:
print("Pre-loading all data.")
all_X = []
all_Y = []
for ds in ['train', 'testA', 'testB']:
    data = WarwickDataFeed(params, ds)
    for idx in range(len(data.files_X)):
        all_X += [data.images_X[idx] / 255.]
        all_Y += [data.images_Y[idx] > 0]

# Pre-load results if it already exists and we only need to update part of it...
import os

all_scores = {}
if (os.path.isfile('results_warwick/all_scores.json')):
    with open('results_warwick/all_scores.json', 'r') as fp:
        all_scores = json.load(fp)

# Getting the results
for clf_name in clfs:
    X_OFFSET = clfs[clf_name]
    print(clf_name, X_OFFSET)

    params["clf_name"] = clf_name

    tf.reset_default_graph()
    sess = tf.Session()
    saver = tf.train.import_meta_graph(os.path.join(CHECKPOINTS_DIR, '%s_best.ckpt.meta' % (clf_name)))
    saver.restore(sess, os.path.join(CHECKPOINTS_DIR, '%s_best.ckpt' % clf_name))

    try:
        X = tf.get_default_graph().get_tensor_by_name("features/X:0")
    except KeyError:
        X = tf.get_default_graph().get_tensor_by_name("ae/features/X:0")

    Y_seg_name = {'F-Norm': 'classifier/segmentation/Relu:0',
                  'F-Norm-2': 'classifier/segmentation/Reshape_1:0',
                  'F-Norm-3': 'classifier/segmentation/BiasAdd:0',
                  'F-Norm-4': 'classifier/segmentation/LeakyRelu/Maximum:0',
                  'F-Norm-5': 'classifier/segmentation/LeakyRelu/Maximum:0',
                  'F-R10': 'classifier/segmentation/LeakyRelu/Maximum:0',
                  'DW-anno': 'output/segmentation_with_sidechain:0'}

    try:
        tensorname = "output/segmentation:0"
        Y_seg = tf.get_default_graph().get_tensor_by_name(tensorname)
    except KeyError:
        tensorname = 'classifier/segmentation/LeakyRelu/Maximum:0'
        lrelu = tf.get_default_graph().get_tensor_by_name(tensorname)
        Y_seg = tf.nn.softmax(lrelu, name='output/segmentation')

    all_pred = []
    all_scores[clf_name] = {'f1': [], 'mcc': [], 'acc': [], 'jacc': []}
    printProgressBar(0, len(all_X))
    for idx in range(len(all_X)):
        im = all_X[idx] - X_OFFSET
        im_anno = all_Y[idx]

        imshape = im_anno.shape
        nr, nc = (overlap * np.ceil((imshape[0] - 1) / tile_size), overlap * np.ceil((imshape[1] - 1) / tile_size))
        yr, xr = (np.arange(0, nr) * ((imshape[0] - 1 - tile_size) // (nr - 1))).astype('int'), (
                    np.arange(0, nc) * ((imshape[1] - 1 - tile_size) // (nc - 1))).astype('int')
        mesh = np.meshgrid(yr, xr)
        tiles = zip(mesh[0].flatten(), mesh[1].flatten())

        im_pred = np.zeros_like(im_anno).astype('float')
        for t in tiles:
            batch_X = [
                im[t[0]:t[0] + tile_size, t[1]:t[1] + tile_size]]  # [im[t[0]:t[0]+tile_size, t[1]:t[1]+tile_size]/255.]
            sm = Y_seg.eval(session=sess, feed_dict={X: batch_X})[:, :, :, 0]
            im_pred[t[0]:t[0] + tile_size, t[1]:t[1] + tile_size] = np.maximum(
                im_pred[t[0]:t[0] + tile_size, t[1]:t[1] + tile_size], sm[0, :, :])

        all_pred += [im_pred]
        mask_pred = im_pred > 0.5

        all_scores[clf_name]['f1'] += [F1(im_anno, mask_pred)]
        all_scores[clf_name]['mcc'] += [MCC(im_anno, mask_pred)]
        all_scores[clf_name]['acc'] += [ACC(im_anno, mask_pred)]
        all_scores[clf_name]['jacc'] += [JACC(im_anno, mask_pred)]
        printProgressBar(idx, len(all_X))

    np.save('e:/data/GlaS/predictions/%s.npy' % clf_name, all_pred)

with open('results_warwick/all_scores.json', 'w') as fp:
    json.dump(all_scores, fp)
