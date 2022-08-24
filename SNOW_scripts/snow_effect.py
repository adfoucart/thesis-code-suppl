'''
Compute the F1-Score of weakened datasets vs the original one, to get an idea of the "SNOW" effects
'''

from data import GlasDataFeed


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


params = {
    "tile_size": 128,
    "dataset_dir": "E:/data/GlaS",
    "annotations": "anno",
    "batch_size": 20,
    "feed_name": "glas",
}

# sets = ['n10', 'n20', 'n30', 'n40', 'n50', 'n60', 'n70', 'n80', 'n90']
# sets = ['bb', 'f20', 'f40', 'f80', 'sig10', 'sig20', 'sig30']
sets = ['noisyhd']
original_feed = GlasDataFeed(params, 'train')

for ds in sets:
    params['annotations'] = ds
    weak_feed = GlasDataFeed(params, 'train')

    f1m = 0
    for i in range(len(original_feed.images_Y)):
        full = original_feed.images_Y[i] > 0
        weak = weak_feed.images_Y[i] > 0

        f1m += F1(full, weak)
    f1m /= len(original_feed.images_Y)
    print(ds, f1m)
