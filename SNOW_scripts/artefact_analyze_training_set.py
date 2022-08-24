'''
Script to get some stats on the Artefact training set:
* % of positive patches ?
* % of positive tissue ?
* Number of annotated objects ?

usage: python artefact_analyze_training_set.py <action> [{<action>}]

<actions> : tissue | patches | objects
'''

import os
from skimage.io import imread
from ArtefactDataFeed import ArtefactDataFeed
from matplotlib import pyplot as plt

def getPercentageAnnotatedTissue():
	fdir = "E:/data/Artefact/slides/train"
	files = os.listdir(fdir)
	files_X = [os.path.join(fdir,f) for f in files if f.find('_rgb') >= 0 and f.find('1.25') >= 0]
	files_B = [os.path.join(fdir,f) for f in files if f.find('_bg') >= 0 and f.find('1.25') >= 0]
	files_Y = [os.path.join(fdir,f) for f in files if f.find('_mask') >= 0 and f.find('1.25') >= 0]

	totArtefact = 0
	totTissue = 0
	for i in range(len(files_X)):
		im = imread(files_X[i])
		imB = imread(files_B[i])==0
		imY = imread(files_Y[i])>0

		totTissue += imB.sum()
		totArtefact += imY.sum()
	return (totArtefact*1./totTissue, totArtefact, totTissue)

def getPercentageAnnotatedPatches():
	params = {
		"tile_size": 128,
		"dataset_dir": "E:/data/Artefact/slides/",
		"annotations": "anno",
		"batch_size": 10,
		"verbose": False
	}

	feed = ArtefactDataFeed(params, 'train')
	tPos = 0
	tTot = 0
	i = 0
	for batch_X,batch_Y_seg,batch_Y_det in feed.next_batch(params['batch_size'], 1):
		tPos += (batch_Y_det[:,0]==1).sum()
		tTot += params['batch_size']
		i += 1

	return (tPos*1./tTot, tPos, tTot)

def getNumberOfAnnotatedObjects():
	from skimage.measure import label
	fdir = "E:/data/Artefact/slides/train"
	files = os.listdir(fdir)
	files_Y = [os.path.join(fdir,f) for f in files if f.find('_mask') >= 0 and f.find('1.25') >= 0]

	nObj = 0
	for f in files_Y:
		objs = label(imread(f)>0)
		print(f, objs.max())
		nObj += objs.max()

	return nObj

def main():
	for i in range(1, len(sys.argv)):
		action = sys.argv[i]
		if( action == 'tissue' ):
			print("Computing %% of annotated tissue:")
			print(getPercentageAnnotatedTissue())
		elif( action == 'patches' ):
			print("Estimating %% of annotated tissue:")
			print(getPercentageAnnotatedPatches())
		elif( action == 'objects' ):
			print("Computing # of annotated objects:")
			print(getNumberOfAnnotatedObjects())
		else:
			print("Unknown action: %s"%action)
			print("usage: python artefact_analyze_training_set.py <action> [{<action>}]")
			print("<actions> : tissue | patches | objects")

if __name__ == '__main__':
	import sys

	if( len(sys.argv) > 1):
		main()
	else:
		print("usage: python artefact_analyze_training_set.py <action> [{<action>}]")
		print("<actions> : tissue | patches | objects")