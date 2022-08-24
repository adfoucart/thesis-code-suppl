'''
22/01/2020

Objective of the script:

* Get some stats about warwick data:
** What is the % of positive vs negative tissue in original set ?
** What is the % of positive vs negative patches ?
** Same in the 50% Noise set ?
'''

from EpitheliumDataFeed import EpitheliumDataFeed
import numpy as np

def getStats(imset):
	pPos = []
	ns = np.array([0,0])

	for im in imset:
		nPos = (im>0).sum()
		nTot = im.shape[0]*im.shape[1]
		ns += np.array([nPos,nTot])
		pPos += [nPos/nTot]

	return pPos, ns

# Print iterations progress
# From Benjamin Cordier https://stackoverflow.com/a/34325723
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '*', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

fp = open('Epithelium Statistics.md', 'w')

# -- Using the original dataset -- #
fp.write("=Original=\n\n")

params = {
	"tile_size": 128,
	"dataset_dir": "E:/data/Epithelium",
	"annotations": "full",
	"batch_size": 10,
	"verbose": False
}

# -- Check in training set -- #
fp.write("==Training Set==\n")
feed = EpitheliumDataFeed(params, 'train')
pPos, ns = getStats(feed.images_Y)
tPos = 0
tTot = 0
i = 0
N_EPOCHS = 1
printProgressBar(0, N_EPOCHS*len(feed.images_Y))
for batch_X,batch_Y_seg,batch_Y_det in feed.next_batch(params['batch_size'], N_EPOCHS):
	tPos += (batch_Y_det[:,0]==1).sum()
	tTot += params['batch_size']
	i += 1
	printProgressBar(i, N_EPOCHS*len(feed.images_Y))

fp.write("\n")
fp.write("* Min/Max %% of object/background pixels in training set: %.2f%% - %.2f%% of pixels\n"%(np.min(pPos)*100, np.max(pPos)*100))
fp.write("* Avg %% of object/background pixels in training set: %.2f%%\n"%(100*ns[0]/ns[1]))
fp.write("* Number of positive tiles: %d\n"%tPos)
fp.write("* Number of possible tiles: %d\n"%tTot)
fp.write("* %% of positive tiles: %.2f%%\n"%(100*tPos/tTot))

# -- Check test set -- #
fp.write("\n")
fp.write("\n==Test Set==\n")
feed = EpitheliumDataFeed(params, 'test')
pPos, ns = getStats(feed.images_Y)
tPos = 0
tTot = 0
i = 0
N_EPOCHS = 1
printProgressBar(0, N_EPOCHS*len(feed.images_Y))
for batch_X,batch_Y_seg,batch_Y_det in feed.next_batch(params['batch_size'], N_EPOCHS):
	tPos += (batch_Y_det[:,0]==1).sum()
	tTot += params['batch_size']
	i += 1
	printProgressBar(i, N_EPOCHS*len(feed.images_Y))

fp.write("\n")
fp.write("* Min/Max %% of object/background pixels in training set: %.2f%% - %.2f%% of pixels\n"%(np.min(pPos)*100, np.max(pPos)*100))
fp.write("* Avg %% of object/background pixels in training set: %.2f%%\n"%(100*ns[0]/ns[1]))
fp.write("* Number of positive tiles: %d\n"%tPos)
fp.write("* Number of possible tiles: %d\n"%tTot)
fp.write("* %% of positive tiles: %.2f%%\n"%(100*tPos/tTot))

# -- Now we do the same on the Noisy set -- #
fp.write("\n=Noisy=\n")
params = {
	"tile_size": 128,
	"dataset_dir": "E:/data/Epithelium",
	"annotations": "n50",
	"batch_size": 10,
	"verbose": False
}

# -- Check in training set -- #
fp.write("\n==Training Set==\n")
feed = EpitheliumDataFeed(params, 'train')
pPos, ns = getStats(feed.images_Y)
tPos = 0
tTot = 0
i = 0
N_EPOCHS = 1
printProgressBar(0, N_EPOCHS*len(feed.images_Y))
for batch_X,batch_Y_seg,batch_Y_det in feed.next_batch(params['batch_size'], N_EPOCHS):
	tPos += (batch_Y_det[:,0]==1).sum()
	tTot += params['batch_size']
	i += 1
	printProgressBar(i, N_EPOCHS*len(feed.images_Y))

fp.write("\n")
fp.write("* Min/Max %% of object/background pixels in training set: %.2f%% - %.2f%% of pixels\n"%(np.min(pPos)*100, np.max(pPos)*100))
fp.write("* Avg %% of object/background pixels in training set: %.2f%%\n"%(100*ns[0]/ns[1]))
fp.write("* Number of positive tiles: %d\n"%tPos)
fp.write("* Number of possible tiles: %d\n"%tTot)
fp.write("* %% of positive tiles: %.2f%%\n"%(100*tPos/tTot))

fp.close()