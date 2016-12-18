import matplotlib.pyplot as plt
import os, glob, ipdb
import configure as cfg
import numpy as np
import cv2

test_ids_list = cfg.PTH_TESTINSTANCE_IDS
data_dir = cfg.DIR_DATA_MASKED
size = 64
max_item = 14
trial = 10

def add_im(category, obj, big, r, c):
    pth = os.path.join(data_dir,category,obj,obj) + '_1_1_crop.png'
    im = cv2.imread(pth, -1)
    im = cv2.resize(im, (size,size), interpolation=cv2.INTER_CUBIC)
    big[r*size:(r+1)*size, c*size:(c+1)*size, :] = im
    return big


if __name__ == '__main__':
    test_ids = open(test_ids_list, 'r').read().splitlines()
    for i in range(len(test_ids)):
        if 'trial '+str(trial) in test_ids[i]:
            test_ids = test_ids[i+1:i+52]
            break

    big = np.zeros((51*size,max_item*size,3))
    r,c,q = 0,0,0

    categories = os.listdir(data_dir)
    categories.sort()
    for category in categories:
        # query object
        big = add_im(category, test_ids[q], big, r, c)
        q += 1; c += 1

        # training objects
        objects = os.listdir(os.path.join(data_dir,category))
        objects.sort()
        for obj in objects:
            if obj in test_ids:
                continue
            big = add_im(category, obj, big, r, c)
            c += 1

        r += 1; c = 0
    cv2.imwrite('trial'+str(trial)+'.png', big)
