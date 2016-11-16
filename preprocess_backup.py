import os, glob, ipdb
import configure as cfg
import numpy as np
import cv2

IMREAD_COLOR = int(cv2.IMREAD_COLOR)
IMREAD_UNCHANGED = int(cv2.IMREAD_UNCHANGED)

def resize_img(img):
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


def colorize_depth(img):
    # scale the value from 0 to 255
    img = img.astype(float)
    img *= 255 / img.max()
    img = img.astype(np.uint8)

    # colorize depth map
    res = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    #res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    #plt.imsave('tmp.png', res)
    return res


def process(dir_input, dir_output):
    # load mean image
    mean_img = np.load(cfg.PTH_MEAN_IMG)
    mean_img = mean_img.transpose(1,2,0)
    mean_img = cv2.resize(mean_img, (cfg.IMG_S,cfg.IMG_S))
    #mean_img = mean_img.transpose(2,0,1)
    mean_img = np.float32(mean_img)

    # go through the whole dataset
    classes = os.listdir(dir_input) # classes
    if '.DS_Store' in classes: classes.remove('.DS_Store')
    if not os.path.exists(dir_output): os.mkdir(dir_output)

    for a_class in classes:
        print a_class
        class_pth = os.path.join(dir_input, a_class)
        objs = os.listdir(class_pth) # objects
        if '.DS_Store' in objs: objs.remove('.DS_Store')
        if not os.path.exists(os.path.join(dir_output, a_class)): os.mkdir(os.path.join(dir_output, a_class))

        for obj in objs:
            print '  ' + obj
            obj_pth = os.path.join(class_pth, obj)
            obj_pth_out = os.path.join(dir_output, a_class, obj)
            if not os.path.exists(obj_pth_out): os.mkdir(obj_pth_out)

            sample_ids = glob.glob1(obj_pth, '*_loc.txt') # samples
            sample_ids = [i.replace('_loc.txt','') for i in sample_ids]
            sample_ids.sort()

            for sid in sample_ids:
                # load images
                rgb = cv2.imread(os.path.join(obj_pth, sid+cfg.EXT_RGB), IMREAD_COLOR)
                dep = cv2.imread(os.path.join(obj_pth, sid+cfg.EXT_D), IMREAD_UNCHANGED)

                # colorize depth
                dep = colorize_depth(dep)

                # resize images
                rgb = resize_img(rgb)
                dep = resize_img(dep)

                # transpose dimension
                #rgb = rgb.transpose(2,0,1)
                #dep = dep.transpose(2,0,1)
                rgb = np.float32(rgb)
                dep = np.float32(dep)

                # mean removal
                rgb -= mean_img
                dep -= mean_img

                # write image
                cv2.imwrite(os.path.join(obj_pth_out, sid+cfg.EXT_RGB), rgb)
                cv2.imwrite(os.path.join(obj_pth_out, sid+cfg.EXT_D), dep)
    return


if __name__ == '__main__':
    #dir_input = cfg.DIR_DATA_RAW
    #dir_output = cfg.DIR_DATA

    dir_input = cfg.DIR_DATA_EVAL_RAW
    dir_output = cfg.DIR_DATA_EVAL

    print 'Input directory: %s' % dir_input
    print 'Output directory: %s' % dir_output
    process(dir_input, dir_output)