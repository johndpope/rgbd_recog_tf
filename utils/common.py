import os, sys, glob, ipdb
import numpy as np
import cv2
import configure as cfg


IMREAD_COLOR = int(cv2.IMREAD_COLOR)
IMREAD_UNCHANGED = int(cv2.IMREAD_UNCHANGED)


def writer(msg, params, f):
    """ print to screen and file
    Args:
        msg: string with format
        params: tuple of parameter
        f: file holder
    """
    print msg % params
    msg += '\n'
    f.write(msg % params)
    return


def write_prob(prob, labels, tag, step):
    f = open(os.path.join(cfg.DIR_PROB, tag+'_'+str(step)+'.txt'), 'a')
    N = len(labels)
    for i in range(N):
        line = prob[i]
        for j in line: f.write('%f ' % j)
        f.write('%d %d\n' % (np.argmax(line), np.argmax(labels[i])))
    f.close()
    return


#batch manager-------------------------------------------------------------------------------------------
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


#data loader----------------------------------------------------------------------------------------------
def parse_label(x, classes):
    return classes.index(x.split('/')[0])


def load_images(lst, data_dir, ext, classes, IMG_S=256):
    # load mean img
    mean_img = np.load(cfg.PTH_MEAN_IMG)
    mean_img = mean_img.transpose(1,2,0)

    N = len(lst)
    images = np.zeros((N,IMG_S,IMG_S,3), dtype=np.uint8)
    labels = np.zeros((N,len(classes)), dtype=np.float32)

    lim = 10
    for i in range(N):
        # read image
        img = cv2.imread(os.path.join(data_dir, lst[i]+ext))

        # mean removal
        img = img.astype(np.float32) - mean_img.astype(np.float32)

        # add to list
        img = img[np.newaxis, ...]
        images[i] = img

        # parse label
        labels[i, parse_label(lst[i], classes)] = 1.0
        
        percent = int(100.0*i/N)
        if percent == lim:
            print '    Loaded %d / %d' % (i, N)
            lim += 10
    return images, labels


from preprocess_4d import resize_dep
def load_4d(lst, rgb_dir, dep_dir, process_dep=False):
    N = len(lst)
    rgbds = np.zeros((N, cfg.IMG_S, cfg.IMG_S, 4), dtype=np.uint8)
    labels = np.zeros((N, len(cfg.CLASSES)), dtype=np.float32)

    lim = 10
    for i in range(N):
        # read rgbd
        rgb = cv2.imread(os.path.join(rgb_dir, lst[i]+cfg.EXT_RGB), IMREAD_COLOR)
        dep = cv2.imread(os.path.join(dep_dir, lst[i]+cfg.EXT_D), IMREAD_UNCHANGED)
        dep = resize_dep(dep).astype(np.float32)

        if process_dep:
            dep = (dep - cfg.DEP_MIN)*1.0 / cfg.DEP_MAX * 255
            dep = dep.astype(np.uint8)
            #l = np.sum((dep>255).astype(np.int32))
            #if l > 0: print l, dep.max() #TODO: remove this line
            dep = np.clip(255, 0, dep)

        rgbd = np.concatenate((rgb, dep[..., np.newaxis]), axis=2)
        rgbds[i] = rgbd[np.newaxis,...]

        #rgbd = np.load(os.path.join(data_dir, lst[i]+cfg.EXT_4D))
        #rgbd[:,:,3] = (rgbd[:,:,3] - cfg.DEP_MIN) / cfg.DEP_MAX * 255.0 # normalize to 0..255
        #rgbd = np.clip(rgbd, 0, 255)
        #rgbds[i] = rgbd[np.newaxis, ...]

        # parse label
        labels[i, parse_label(lst[i], cfg.CLASSES)] = 1.0

        # check progress
        percent = int(100.0 * i / N)
        if percent == lim:
            print '    Loaded %d / %d' %(i, N)
            lim += 10
        
    return rgbds, labels


#image helpers-----------------------------------------------------------------------------------------
def random_crop(images, rand_fl=False):
    """ Randomly crop the whole batch of image with the same mask
    """
    old_size = images.shape[1]
    new_size = cfg.IMG_S
    r = old_size - new_size
    u = np.random.randint(r+1)
    v = np.random.randint(r+1)
    images = images[:, u:new_size+u, v:new_size+v, :]
    if rand_fl:
        return random_flip(images)
    return images


def central_crop(images):
    old_size = images.shape[1]
    new_size = cfg.IMG_S
    r = (old_size-new_size)/2
    images = images[:, r:new_size+r, r:new_size+r, :]
    return images


def random_flip(images):
    ids = np.random.choice(len(images), len(images)/20)
    images[ids,:,:,:] = images[ids,:,::-1,:]
    return images
