import os, glob, ipdb
import configure as cfg
import numpy as np
import cv2
import configure as cfg


IMREAD_COLOR = int(cv2.IMREAD_COLOR)
IMREAD_UNCHANGED = int(cv2.IMREAD_UNCHANGED)


def resize_dep(img):
    def _expand_top_bottom(img):
        im_h,im_w = img.shape
        new_h = int(im_h*cfg.IMG_S/im_w)
        img = cv2.resize(img, (cfg.IMG_S,new_h))

        # get number of top and bottom rows
        nb_top = int((cfg.IMG_S-new_h)/2)
        nb_bottom = cfg.IMG_S-new_h-nb_top

        # duplicate top and bottom rows
        res = img
        if nb_top!=0:
            top = np.array([img[0,:]]*nb_top)
            res = np.concatenate((top,res), axis=0)
        if nb_bottom!=0:
            bottom = np.array([img[-1,:]]*nb_bottom)
            res = np.concatenate((res,bottom), axis=0)
        return res

    im_h,im_w = img.shape
    if im_w == im_h:
        return cv2.resize(img, (cfg.IMG_S, cfg.IMG_S))
    if im_w>im_h:
        res = _expand_top_bottom(img)
    else:
        res = _expand_top_bottom(img.T).T

    return res


def resize_rgb(img):
    im_h,im_w,im_c = img.shape

    if im_w>im_h:
        new_h = int(im_h*cfg.IMG_S/im_w)
        img = cv2.resize(img, (cfg.IMG_S,new_h))

        # get number of top and bottom rows
        nb_top = int((cfg.IMG_S-new_h)/2)
        nb_bottom = cfg.IMG_S-new_h-nb_top

        # duplicate top and bottom rows
        if nb_top!=0:
            top = np.dstack([img[0,:,:]]*nb_top)
            top = top.transpose(2,0,1)
        if nb_bottom!=0:
            bottom = np.dstack([img[-1,:,:]]*nb_bottom)
            bottom = bottom.transpose(2,0,1)

        # concatenate with the original image
        if nb_top==0 and nb_bottom!=0:
            res = np.concatenate((img,bottom), axis=0)
        elif nb_top!=0 and nb_bottom==0:
            res = np.concatenate((top,img), axis=0)
        else:
            res = np.concatenate((top,img,bottom), axis=0)
    elif im_w<im_h:
        new_w = int(im_w*cfg.IMG_S/im_h)
        img = cv2.resize(img, (new_w,cfg.IMG_S))

        # get number of left and right cols
        nb_left = int((cfg.IMG_S-new_w)/2)
        nb_right = cfg.IMG_S-new_w-nb_left

        # duplicate left and right cols
        if nb_left!=0:
            left = np.dstack([img[:,0,:]]*nb_left)
            left = left.transpose(0,2,1)
        if nb_right!=0:
            right = np.dstack([img[:,-1,:]]*nb_right)
            right = right.transpose(0,2,1)

        # concatenate with the original image
        if nb_left==0 and nb_right!=0:
            res = np.concatenate((img,right), axis=1)
        elif nb_left!=0 and nb_right==0:
            res = np.concatenate((left,img), axis=1)
        else:
            res = np.concatenate((left,img,right), axis=1)
    else:
        res = cv2.resize(img, (cfg.IMG_S,cfg.IMG_S))
    return res


def resize():
    # go through the whole dataset
    classes = os.listdir(cfg.DIR_DATA_RAW) # classes
    if '.DS_Store' in classes: classes.remove('.DS_Store')
    if not os.path.exists(cfg.DIR_DATA_4D): os.mkdir(cfg.DIR_DATA_4D)

    for a_class in classes:
        print a_class
        class_pth = os.path.join(cfg.DIR_DATA_RAW, a_class)
        objs = os.listdir(class_pth) # objects
        if '.DS_Store' in objs: objs.remove('.DS_Store')
        if not os.path.exists(os.path.join(cfg.DIR_DATA_4D, a_class)): os.mkdir(os.path.join(cfg.DIR_DATA_4D, a_class))

        for obj in objs:
            print '    '+obj
            obj_pth = os.path.join(class_pth, obj)
            obj_pth_out = os.path.join(cfg.DIR_DATA_4D, a_class, obj)
            if not os.path.exists(obj_pth_out): os.mkdir(obj_pth_out)

            sample_ids = glob.glob1(obj_pth, '*_loc.txt') # samples
            sample_ids = [i.replace('_loc.txt','') for i in sample_ids]
            sample_ids.sort()

            for sid in sample_ids:
                # load images
                rgb = cv2.imread(os.path.join(obj_pth, sid+cfg.EXT_RGB), IMREAD_COLOR)
                dep = cv2.imread(os.path.join(obj_pth, sid+cfg.EXT_D), IMREAD_UNCHANGED)

                # resize images
                rgb = resize_rgb(rgb)
                dep = resize_dep(dep)
                rgb = np.float32(rgb)
                dep = np.float32(dep)

                # write image
                rgbd = np.concatenate((rgb,dep[..., np.newaxis]), axis=2)
                np.save(os.path.join(obj_pth_out, sid+cfg.EXT_4D), rgbd)
    return


def analyze():
    with open(cfg.PTH_TRAIN_LST,'r') as f: train_lst = f.read().splitlines()
    N = len(train_lst)

    '''
    # max depth
    import sys
    dep_min, dep_max = sys.maxsize, 0.0
    for i in range(N):
        rgbd = np.load(os.path.join(cfg.DIR_DATA_4D, train_lst[i]+cfg.EXT_4D))
        dep_min = min(dep_min, rgbd[:,:,3].min())
        dep_max = max(dep_max, rgbd[:,:,3].max())
    print dep_min, dep_max
    '''

    # mean
    if os.path.exists(cfg.PTH_RGBD_MEAN):
        rgbd_mean = np.load(cfg.PTH_RGBD_MEAN)
    else:
        rgbd_mean = np.zeros([cfg.IMG_S, cfg.IMG_S, 4])
        for i in range(N):
            rgbd = np.load(os.path.join(cfg.DIR_DATA_4D, train_lst[i]+cfg.EXT_4D))
            rgbd_mean += rgbd.astype(np.float32) / N
        np.save(cfg.PTH_RGBD_MEAN, rgbd_mean)
    
    # variance
    rgbd_var = np.zeros([cfg.IMG_S, cfg.IMG_S, 4])
    for i in range(N):
        rgbd = np.load(os.path.join(cfg.DIR_DATA_4D, train_lst[i]+cfg.EXT_4D))
        rgbd_var += (rgbd.astype(np.float32) - rgbd_mean)**2 / N
    np.save(cfg.PTH_RGBD_VAR, rgbd_var)
    return 


if __name__ == '__main__':
    resize()
    analyze()
