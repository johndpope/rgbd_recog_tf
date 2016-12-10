import os, ipdb
import tensorflow as tf
import numpy as np
import configure as cfg
from utils import common
from architectures.tfcommon import conv


def extract(images, model, prefix, batch_size):
    # conv-1 layer
    with tf.name_scope(prefix+'conv1') as scope:
        conv1W = tf.Variable(model[prefix+'conv1W'], trainable=False, name='weight')
        conv1b = tf.Variable(model[prefix+'conv1b'], trainable=False, name='biases')
        conv1_in = conv(images, conv1W, conv1b, 11, 11, 96, 4, 4, padding='SAME', group=1)
        conv1 = tf.nn.relu(conv1_in, name=scope)
    maxpool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name=prefix+'pool1')
    lrn1 = tf.nn.local_response_normalization(maxpool1, depth_radius=5, alpha=1e-4, beta=0.75, name=prefix+'norm1')


    # conv-2 layer
    with tf.name_scope(prefix+'conv2') as scope:
        conv2W = tf.Variable(model[prefix+'conv2W'], trainable=False, name='weight')
        conv2b = tf.Variable(model[prefix+'conv2b'], trainable=False, name='biases')
        conv2_in = conv(maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding='SAME', group=2)
        conv2 = tf.nn.relu(conv2_in, name=scope)
    maxpool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name=prefix+'pool2')
    lrn2 = tf.nn.local_response_normalization(maxpool2, depth_radius=5, alpha=1e-4, beta=0.75, name=prefix+'norm2')


    # conv-3 layer
    with tf.name_scope(prefix+'conv3') as scope:
        conv3W = tf.Variable(model[prefix+'conv3W'], trainable=False, name='weight')
        conv3b = tf.Variable(model[prefix+'conv3b'], trainable=False, name='biases')
        conv3_in = conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding='SAME', group=1)
        conv3 = tf.nn.relu(conv3_in, name=scope)


    # conv-4 layer
    with tf.name_scope(prefix+'conv4') as scope:
        conv4W = tf.Variable(model[prefix+'conv4W'], trainable=False, name='weight')
        conv4b = tf.Variable(model[prefix+'conv4b'], trainable=False, name='biases')
        conv4_in = conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding='SAME', group=2)
        conv4 = tf.nn.relu(conv4_in, name=scope)


    # conv-5 layer
    with tf.name_scope(prefix+'conv5') as scope:
        conv5W = tf.Variable(model[prefix+'conv5W'], trainable=False, name='weight')
        conv5b = tf.Variable(model[prefix+'conv5b'], trainable=False, name='biases')
        conv5_in = conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding='SAME', group=2)
        conv5 = tf.nn.relu(conv5_in, name=scope)
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name=prefix+'pool5')


    # fc6 layer
    with tf.name_scope(prefix+'fc6') as scope:
        fc6W = tf.Variable(model[prefix+'fc6W'], trainable=False, name='weight')
        fc6b = tf.Variable(model[prefix+'fc6b'], trainable=False, name='biases')
        fc6_in = tf.reshape(maxpool5, [batch_size, int(np.prod(maxpool5.get_shape()[1:]))])
        fc6 = tf.nn.relu_layer(fc6_in, fc6W, fc6b, name=scope)
        #fc6_drop = tf.nn.dropout(fc6, keep_prob=1.0, name='drop')


    # fc7 layer
    with tf.name_scope(prefix+'fc7') as scope:
        fc7W = tf.Variable(model[prefix+'fc7W'], trainable=False, name='weight')
        fc7b = tf.Variable(model[prefix+'fc7b'], trainable=False, name='biases')
        fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b, name=scope)
        #fc7_drop = tf.nn.dropout(fc7, keep_prob=1.0, name='drop')
                
    feature = fc7
    return feature


def caller(images, model, tag, batch_size=1):
    sess = tf.InteractiveSession()
    img_ph = tf.placeholder(tf.float32, shape=(batch_size,cfg.IMG_S,cfg.IMG_S,3),name='images_placeholder')
    extractor = extract(img_ph, model, tag+'_', batch_size)
    sess.run(tf.initialize_all_variables())

    N=len(images)
    all_feat = np.zeros((N,4096),dtype=np.float32)
    for i in range(0,N,batch_size):
        batch = images[i:i+batch_size]
        feat = sess.run(extractor, feed_dict={img_ph:batch})
        all_feat[i:i+batch_size] = feat
    return all_feat


if __name__ == '__main__':
    # load data
    rgb_model = np.load(cfg.PTH_RGB_MODEL).item()
    dep_model = np.load(cfg.PTH_DEP_MODEL).item()
    lst = open(cfg.PTH_FULL_LST, 'r').read().splitlines()
    N = len(lst)

    # setup
    batch_size = 1
    sess = tf.InteractiveSession()
    img_ph = tf.placeholder(tf.float32, shape=(batch_size,cfg.IMG_S,cfg.IMG_S,3),name='images_placeholder')
    rgb_extractor = extract(img_ph, rgb_model, 'rgb_', batch_size)
    dep_extractor = extract(img_ph, dep_model, 'dep_', batch_size)
    sess.run(tf.initialize_all_variables())

    # extract rgb features
    for i in range(0,N,batch_size):
        batch = lst[i:i+batch_size]
        rgb,dep,_ = common.load_pairs(batch, cfg.DIR_DATA_MASKED, cfg.CLASSES, crop='random')
        rgb_feat = sess.run(rgb_extractor, feed_dict={img_ph:rgb})
        dep_feat = sess.run(dep_extractor, feed_dict={img_ph:dep})

        for item in batch:
            foo,bar = os.path.split(item)
            pth = os.path.join(cfg.DIR_DATA_MASKED_FEAT, foo)
            if not os.path.exists(pth): os.makedirs(pth)
            np.save(os.path.join(pth,bar+cfg.EXT_RGB_FEAT), rgb_feat)
            np.save(os.path.join(pth,bar+cfg.EXT_D_FEAT), dep_feat)

