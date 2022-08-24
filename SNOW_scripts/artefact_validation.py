import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from ArtefactDataFeed import ArtefactDataFeed
from skimage.io import imread, imsave
from skimage.morphology import opening, closing, disk
from sklearn.metrics import auc
from dhutils.tools import F1, MCC

def accuracy(T,P):
	return (T==P).sum()/((T==P).sum()+(T!=P).sum())

def ROC_AUC(T,P):
	n_points = 11
	roc = np.zeros((n_points,2))
	for i in range(n_points):
		t = i/(n_points-1)
		pred_b,gt_ = P>=t,T

		tp = (gt_*pred_b).sum()
		tn = ((gt_==False)*(pred_b==False)).sum()
		fp = ((gt_==False)*pred_b).sum()
		fn = (gt_*(pred_b==False)).sum()

		roc[i,0] = tp/(tp+fn)
		roc[i,1] = 1-(tn/(tn+fp))
	rocauc = auc(roc[:,1], roc[:,0])

	return rocauc

clfs = ['shortres-artefact-noisysw']
X_OFFSET = 0.5 # Use 0 for older networks, 0.5 for networks trained on 0-centered data.

for clf_name in clfs:
	print(clf_name)

	params = {
		"clf_name": clf_name,
		"tile_size": 128,
		"summaries_dir": "e:/data/tf_summaries/artefact",
		"checkpoints_dir": "e:/data/tf_checkpoint/artefact",
		"dataset_dir": "e:/data/Artefact/slides"
	}
	CHECKPOINTS_DIR = params['checkpoints_dir']
	overlap = 2
	tile_size = params['tile_size']

	tf.reset_default_graph()
	sess = tf.Session()
	saver = tf.train.import_meta_graph(os.path.join(CHECKPOINTS_DIR,'%s_best.ckpt.meta'%(clf_name)))
	saver.restore(sess, os.path.join(CHECKPOINTS_DIR, '%s_best.ckpt'%clf_name))

	try:
		X = tf.get_default_graph().get_tensor_by_name("features/X:0")
	except KeyError:
		X = tf.get_default_graph().get_tensor_by_name("ae/features/X:0")

	try:
		tensorname = "output/segmentation:0"
		Y_seg = tf.get_default_graph().get_tensor_by_name(tensorname)
	except KeyError:
		print("Couldn't find output tensor")
		break

	f1s = []
	aucs = []

	for ds in ['validation', 'test']:

		data = ArtefactDataFeed(params, ds)

		for idx in range(len(data.files_X)):
			im = data.images_X[idx]/255. - X_OFFSET#imread(data.files_X[idx])
			im_anno = data.images_Y[idx]>0#imread(data.files_Y[idx]) > 0

			imshape = im_anno.shape
			nr,nc = (overlap*np.ceil((imshape[0]-1)/tile_size), overlap*np.ceil((imshape[1]-1)/tile_size))
			yr,xr = (np.arange(0, nr)*((imshape[0]-1-tile_size)//(nr-1))).astype('int'), (np.arange(0, nc)*((imshape[1]-1-tile_size)//(nc-1))).astype('int')
			mesh = np.meshgrid(yr,xr)
			tiles = zip(mesh[0].flatten(), mesh[1].flatten())

			im_pred = np.zeros_like(im_anno).astype('float')
			for t in tiles:
				batch_X = [im[t[0]:t[0]+tile_size, t[1]:t[1]+tile_size]]
				sm = Y_seg.eval(session=sess, feed_dict={X:batch_X})[:,:,:,0]
				im_pred[t[0]:t[0]+tile_size, t[1]:t[1]+tile_size] = np.maximum(im_pred[t[0]:t[0]+tile_size, t[1]:t[1]+tile_size], sm[0,:,:])

			mask_pred = im_pred>0.5
			
			imsave(os.path.join('results_artefact', '%s_%s_proba.png'%(clf_name, data.files_X[idx].rsplit('\\')[-1])), im_pred)
			imsave(os.path.join('results_artefact', '%s_%s_mask.png'%(clf_name, data.files_X[idx].rsplit('\\')[-1])), mask_pred)

			# f1 = F1(im_anno, mask_pred)
			# roc_auc = ROC_AUC(im_anno, im_pred)

			# f1s += [("%f\n"%f1).replace('.',',')]
			# accs += [("%f\n"%acc).replace('.',',')]
			# aucs += [("%f\n"%roc_auc).replace('.',',')]

			# print("%s %d\t %f"%(ds, idx+1, dice))
			#print("%f"%dice)

	# with open('results_warwick/f1%s/res-f1-%s.txt'%(resdirapp,clf_name), 'w') as fp:
	# 	fp.writelines(f1s)
	# with open('results_warwick/auc/res-auc-%s.txt'%clf_name, 'w') as fp:
	# 	fp.writelines(aucs)