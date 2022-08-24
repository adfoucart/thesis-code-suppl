import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from WarwickDataFeed import WarwickDataFeed
from skimage.io import imread
from skimage.morphology import opening, closing, disk
from sklearn.metrics import auc

def softmax2(k):
	e = np.exp(k-k.max())
	s = np.exp(k-k.max()).sum(axis=-1)
	s[s==0] = 1
	return e[...,0]/s

def Dice(T,P):
	TP = ((T==1)*(P==1)).sum()
	FP = ((T==0)*(P==1)).sum()
	return TP/(T.sum()+FP)

def F1(T,P):
	TP = ((T==1)*(P==1)).sum()
	TN = ((T==0)*(P==0)).sum()
	FP = ((T==0)*(P==1)).sum()
	FN = ((T==1)*(P==0)).sum()

	recall = TP/(TP+FN) if TP+FN > 0 else 0
	precision = TP/(TP+FP) if TP+FP > 0 else 0
	if( (recall + precision) == 0 ): return 0
	return 2*precision*recall/(precision+recall)

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


#clfs = ['F-Norm-4', 'F-R10', 'F-R20', 'F-R30', 'F-R40', 'F-R50', 'F-LD', 'F-MD', 'F-HD', 'F-BB', 'F-R10-LD', 'F-R30-MD', 'F-R50-HD', 'F-R50-BB', 'SS-anno']
#clfs = ['SS-R50', 'SS-HD', 'F-R60', 'F-R70', 'F-R80']
#clfs = ['SSP-R10', 'SSP-R20', 'SSP-R30', 'SSP-R40', 'SSP-R60', 'SSP-R70', 'SSP-R80']
#clfs = ['F-BB-B10', 'F-BB-B20', 'GEN-BB-B10', 'GEN-BB-B20', 'NWi-BB-B10', 'NWi-BB-B20', 'SS0-BB-B10', 'SS0-BB-B20', 'SSP-BB-B10', 'SSP-BB-B20', 'SSNWi-BB-B10', 'SSNWi-BB-B20', 'SSWi-BB-B10', 'SSWi-BB-B20', 'W-BB-B10', 'W-BB-B20', 'Wi-BB-B10', 'Wi-BB-B20']
# clfs = ['F-Norm-4', 'F-R50', 'F-BB', 'F-R50-BB', 'F-R50-HD', 
# 		'GEN-anno', 'GEN-R50', 'GEN-BB', 'GEN-R50-BB', 'GEN-R50-HD', 
# 		'SSP-anno', 'SSP-r50', 'SSP-BB', 'SSP-R50-BB', 'SSP-R50-HD', 
# 		'Wmax-anno', 'W-R50', 'W-BB', 'W-R50-BB', 'W-R50-HD', 
# 		'Wi-anno', 'Wi-R50', 'Wi-BB', 'Wi-R50-BB', 'Wi-R50-HD', 
# 		'NWi-anno', 'NWi-R50', 'NWi-BB', 'NWi-R50-BB', 'NWi-R50-HD', 
# 		'SSWi-anno', 'SSWi-R50', 'SSWi-BB', 'SSWi-R50-BB', 'SSWi-R50-HD', 
# 		'SSNWi-anno', 'SSNWi-R50', 'SSNWi-BB', 'SSNWi-R50-BB', 'SSNWi-R50-HD', 
# 		'SS0-anno', 'SS0-R50', 'SS0-BB', 'SS0-R50-BB', 'SS0-R50-HD']
#clfs = ['GSN75-r50-datacentered','GSN75-bb-datacentered','GSN75-r50-bb-datacentered','GSN75-r50-hd-datacentered']
#clfs = ['GEN-r50-datacentered', 'GSN100-r50-datacentered']
clfs = ['F-Norm-4']
X_OFFSET = 0. # Use 0 for older networks, 0.5 for networks trained on 0-centered data.

WITH_MORPHOLOGY = False
resdirapp = '-pp' if WITH_MORPHOLOGY else ''

for clf_name in clfs:
	#clf_name = "F-R10"
	print(clf_name)

	params = {
		"clf_name": clf_name,
		"tile_size": 256,
		"summaries_dir": "e:/data/tf_summaries/warwick_snow",
		"checkpoints_dir": "e:/data/tf_checkpoint/warwick_snow",
		# "summaries_dir": "e:/data/tf_summaries/pan",
		# "checkpoints_dir": "e:/data/tf_checkpoint/pan",
		"dataset_dir": "e:/data/GlaS",
		"annotations": "anno"
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

	Y_seg_name = { 'F-Norm': 'classifier/segmentation/Relu:0',
				   'F-Norm-2': 'classifier/segmentation/Reshape_1:0',
				   'F-Norm-3': 'classifier/segmentation/BiasAdd:0',
				   'F-Norm-4': 'classifier/segmentation/LeakyRelu/Maximum:0',
				   'F-Norm-5': 'classifier/segmentation/LeakyRelu/Maximum:0',
				   'F-R10': 'classifier/segmentation/LeakyRelu/Maximum:0',
				   'DW-anno': 'output/segmentation_with_sidechain:0'}

	#tensorname = Y_seg_name[clf_name] if clf_name in Y_seg_name else 'classifier/segmentation/LeakyRelu/Maximum:0'
	# New and better :
	try:
		tensorname = "output/segmentation:0"
		Y_seg = tf.get_default_graph().get_tensor_by_name(tensorname)
	except KeyError:
		tensorname = Y_seg_name[clf_name] if clf_name in Y_seg_name else 'classifier/segmentation/LeakyRelu/Maximum:0'
		lrelu = tf.get_default_graph().get_tensor_by_name(tensorname)
		Y_seg = tf.nn.softmax(lrelu, name='output/segmentation')

	f1s = []
	accs = []
	aucs = []

	for ds in ['train', 'testA', 'testB']:

		data = WarwickDataFeed(params, ds)

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
				batch_X = [im[t[0]:t[0]+tile_size, t[1]:t[1]+tile_size]]#[im[t[0]:t[0]+tile_size, t[1]:t[1]+tile_size]/255.]
				sm = Y_seg.eval(session=sess, feed_dict={X:batch_X})[:,:,:,0]
				# pred = Y_seg.eval(session=sess, feed_dict={X:batch_X})
				#if( clf_name != 'F-Norm-2' ):
				#	sm = softmax2(pred)
				#else:
				#	sm = pred[:,:,:,0]
				im_pred[t[0]:t[0]+tile_size, t[1]:t[1]+tile_size] = np.maximum(im_pred[t[0]:t[0]+tile_size, t[1]:t[1]+tile_size], sm[0,:,:])

			mask_pred = im_pred>0.5
			if( WITH_MORPHOLOGY ):
				mask_pred = closing(opening(mask_pred, disk(2)), disk(10))

			f1 = F1(im_anno, mask_pred)
			acc = accuracy(im_anno, mask_pred)
			roc_auc = ROC_AUC(im_anno, im_pred)

			f1s += [("%f\n"%f1).replace('.',',')]
			accs += [("%f\n"%acc).replace('.',',')]
			aucs += [("%f\n"%roc_auc).replace('.',',')]

			# print("%s %d\t %f"%(ds, idx+1, dice))
			#print("%f"%dice)

	with open('results_warwick/f1-check%s/res-f1-%s.txt'%(resdirapp,clf_name), 'w') as fp:
		fp.writelines(f1s)
	# with open('results_warwick/acc/res-acc-%s.txt'%clf_name, 'w') as fp:
	# 	fp.writelines(accs)
	# with open('results_warwick/auc/res-auc-%s.txt'%clf_name, 'w') as fp:
	# 	fp.writelines(aucs)