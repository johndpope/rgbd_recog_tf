import tensorflow as tf
import numpy as np
import os, sys, ipdb
import matplotlib.pyplot as plt


ckpt_dir = '../checkpoints'
ckpt_id = 'dep-42'


def visualize_weight(weight, title=''):
    spacing = 1
    hei,wid,inp,out = weight.shape
    im = np.zeros( (hei*inp + spacing*(inp-1), wid*out + spacing*(out-1)) )

    r, u = 0, 0
    while u < inp:
        c, v = 0, 0
        while v < out:
            im[r:r+hei, c:c+wid] = weight[:,:,u,v]
            c += wid + spacing 
            v += 1
        r += hei + spacing
        u += 1
    im = np.dstack( (im,im,im) )
    plt.figure()
    plt.imshow(im)
    plt.title(title)
    return im


def main(path):
    sess = tf.Session()

    # load checkpoint
    saver = tf.train.import_meta_graph(path+'.meta')
    saver.restore(sess, path)
    var_lst = tf.trainable_variables()
    for v in var_lst:
        print v.op.name

    # load variables' values
    model = dict()
    model['conv1W'] = sess.run(var_lst[0])
    model['conv1b'] = sess.run(var_lst[1])
    model['conv2W'] = sess.run(var_lst[2])
    model['conv2b'] = sess.run(var_lst[3])
    model['conv3W'] = sess.run(var_lst[4])
    model['conv3b'] = sess.run(var_lst[5])
    model['conv4W'] = sess.run(var_lst[6])
    model['conv4b'] = sess.run(var_lst[7])
    model['conv5W'] = sess.run(var_lst[8])
    model['conv5b'] = sess.run(var_lst[9])
    model['fc6W'] = sess.run(var_lst[10])
    model['fc6b'] = sess.run(var_lst[11])
    model['fc7W'] = sess.run(var_lst[12])
    model['fc7b'] = sess.run(var_lst[13])
    model['fc8W'] = sess.run(var_lst[14])
    model['fc8b'] = sess.run(var_lst[15])

    # draw conv1 weight as color
    hei,wid,inp,out = model['conv1W'].shape
    tiles = int(np.ceil(np.sqrt(out)))
    im1 = np.zeros((hei*tiles, wid*tiles,inp))
    idx = 0
    for r in range(tiles):
        for c in range(tiles):
            if idx >= out:
                break
            im1[r*hei:(r+1)*hei, c*wid:(c+1)*wid] = model['conv1W'][:,:,:,idx]
            idx += 1
    plt.figure()
    plt.imshow(im1)
    plt.title('conv1W as color')

    # draw weight wrt input (row) and output (column) dimension
    visualize_weight(model['conv1W'], 'conv1W')
    visualize_weight(model['conv2W'], 'conv2W')
    visualize_weight(model['conv3W'], 'conv3W')
    visualize_weight(model['conv4W'], 'conv4W')
    visualize_weight(model['conv5W'], 'conv5W')

    plt.show()

    return


if __name__ == '__main__':
    with tf.Graph().as_default():
        main(sys.argv[1:])
