import os, sys, glob, ipdb
import numpy as np
import cv2


def early_stopping(old_val, new_val, patience_count, tolerance=1e-2, patience_limit=3):
    to_stop = False
    improvement = new_val - old_val
    if improvement < tolerance:
        if patience_count < patience_limit:
            patience_count += 1
        else:
            to_stop = True
    else:
        patience_count = 0
    return to_stop, patience_count


def load_images(lst, data_dir, ext, classes):
    images = []
    labels = []
    for i in lst:
        img = cv2.imread(os.path.join(data_dir, i+ext))
        img = img[np.newaxis, ...]
        if images == []:
            images = img
        else:
            images = np.concatenate((images, img), axis=0)

        foo = classes.index(i.split('/')[0])
        bar = np.zeros((1,len(classes)))
        bar[0,foo] = 1.0
        if labels == []:
            labels = bar
        else:
            labels = np.concatenate((labels, bar), axis=0)
    return images, labels
