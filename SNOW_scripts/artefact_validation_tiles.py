import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from ArtefactDataFeed import ArtefactDataFeed
from skimage.io import imread, imsave

# clfs = ['pan-artefact-baseline', 'pan-artefact-onlyP', 'pan-artefact-gsn50', 'pan-artefact-sw', 'pan-artefact-sssw']
# clfs = ['unet-artefact-gsn50']
clfs = ['shortres-artefact-weak']
X_OFFSET = 0.5 # Use 0 for older networks, 0.5 for networks trained on 0-centered data.

for clf_name in clfs:
	print(clf_name)

	params = {
		"clf_name": clf_name,
		"tile_size": 128,
		"summaries_dir": "e:/data/tf_summaries/artefact",
		"checkpoints_dir": "e:/data/tf_checkpoint/artefact",
		"dataset_dir": "e:/data/Artefact/slides",
		"removeBackground": False
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

	for ds in ['validation_tiles', 'test']:

		data = ArtefactDataFeed(params, ds)

		for idx in range(len(data.files_X)):
			im = data.images_X[idx]/255. - X_OFFSET#imread(data.files_X[idx])
			
			imshape = im.shape
			nr,nc = (overlap*np.ceil((imshape[0]-1)/tile_size), overlap*np.ceil((imshape[1]-1)/tile_size))
			yr,xr = (np.arange(0, nr)*((imshape[0]-1-tile_size)//(nr-1))).astype('int'), (np.arange(0, nc)*((imshape[1]-1-tile_size)//(nc-1))).astype('int')
			mesh = np.meshgrid(yr,xr)
			tiles = zip(mesh[0].flatten(), mesh[1].flatten())

			im_pred = np.zeros(imshape[:2]).astype('float')
			for t in tiles:
				batch_X = [im[t[0]:t[0]+tile_size, t[1]:t[1]+tile_size]]
				sm = Y_seg.eval(session=sess, feed_dict={X:batch_X})[:,:,:,0]
				im_pred[t[0]:t[0]+tile_size, t[1]:t[1]+tile_size] = np.maximum(im_pred[t[0]:t[0]+tile_size, t[1]:t[1]+tile_size], sm[0,:,:])

			mask_pred = im_pred>0.5

			im_final = data.images_X[idx].copy()
			im_final[mask_pred,0] = 255
			
			if( os.path.isdir('results_artefact/%s'%clf_name) == False ):
				os.mkdir('results_artefact/%s'%clf_name)
			imsave(os.path.join('results_artefact/%s'%clf_name, '%s_proba.png'%(data.files_X[idx].rsplit('\\')[-1])), im_pred)
			imsave(os.path.join('results_artefact/%s'%clf_name, '%s_final.png'%(data.files_X[idx].rsplit('\\')[-1])), im_final)

			