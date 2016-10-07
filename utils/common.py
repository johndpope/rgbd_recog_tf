import os, sys, glob, ipdb
import numpy as np
import cv2


def next_batch(indices, start_idx, batch_size):
    N = indices.shape[0]
    if start_idx+batch_size > N:
        stop_idx = N
    else:
        stop_idx = start_idx+batch_size
    return stop_idx


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
    N = len(lst)
    IMG_S = 227 # TODO: make it flexible
    images = np.zeros((N,IMG_S,IMG_S,3), dtype=np.uint8)
    labels = np.zeros((N,len(classes)), dtype=np.float32)

    lim = 10
    for i in range(N):
        # read image
        img = cv2.imread(os.path.join(data_dir, lst[i]+ext))
        img = img[np.newaxis, ...]
        images[i] = img

        # parse label
        loc = classes.index(lst[i].split('/')[0])
        labels[i,loc] = 1.0
        
        percent = int(100.0*i/N)
        if percent == lim:
            print '    Loaded %d / %d' % (i, N)
            lim += 10
    return images, labels
