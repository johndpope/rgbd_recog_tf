import os, ipdb
import tensorflow as tf
import numpy as np
import configure as cfg
from tfcommon import *

FLAGS = tf.app.flags.FLAGS

def _extract_feature(images, model, keep_prob, prefix, batch_size):
    ''' Extract feature using a pretrained weights from single channel model

    Args:
        images: 4D tensor: [batch_size, image_size, image_size, 3]
        model: pretrained single_channel_model (either color or depth)
        keep_prob: keeping probality for training
        prefix: 'rgb_' or 'dep_'
        batch_size: size of each batch

    Returns:
        feature: extracted feature
    '''
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
        fc6_drop = tf.nn.dropout(fc6, keep_prob=keep_prob, name='drop')


    # fc7 layer
    with tf.name_scope(prefix+'fc7') as scope:
        fc7W = tf.Variable(model[prefix+'fc7W'], trainable=False, name='weight')
        fc7b = tf.Variable(model[prefix+'fc7b'], trainable=False, name='biases')
        fc7 = tf.nn.relu_layer(fc6_drop, fc7W, fc7b, name=scope)
        fc7_drop = tf.nn.dropout(fc7, keep_prob=keep_prob, name='drop')
                
    feature = fc7_drop
    return feature


def inference(rgb_img, dep_img, rgb_model, dep_model, keep_prob, tag='fus'):
    tag += '_'
    batch_size = FLAGS.batch_size
    n_classes = FLAGS.n_classes
    
    rgb_feat = _extract_feature(rgb_img, rgb_model, keep_prob, 'rgb_', batch_size)
    dep_feat = _extract_feature(dep_img, dep_model, keep_prob, 'dep_', batch_size)

    concat_feat = tf.concat(1, [rgb_feat, dep_feat], name='concat_feat')
    #concat_W = np.concatenate((rgb_model['rgb_fc8W'], dep_model['dep_fc8W']))
    #concat_b = np.concatenate((rgb_model['rgb_fc8b'], dep_model['dep_fc8b']))

    # fc1-fus
    with tf.name_scope(tag+'fc1_fus') as scope:
        #fc1_fusW = tf.Variable(tf.random_normal([4096*2,4096], mean=concat_W.mean(), stddev=concat_W.std()), name='weight')
        #fc1_fusb = tf.Variable(tf.random_normal([4096], mean=concat_b.mean(), stddev=concat_b.std()), name='biases')
        fc1_fusW = tf.Variable(tf.random_normal([4096*2,4096], stddev=0.01), name='weight')
        fc1_fusb = tf.Variable(tf.zeros([4096]), name='biases')
        fc1_fus  = tf.nn.relu_layer(concat_feat, fc1_fusW, fc1_fusb, name=scope)

    # classifier
    with tf.name_scope(tag+'class') as scope:
        #classW = tf.Variable(tf.random_normal([4096,n_classes], mean=concat_W.mean(), stddev=concat_W.std()), name='weight')
        #classb = tf.Variable(tf.random_normal([n_classes], mean=concat_b.mean(), stddev=concat_b.std()), name='biases')
        classW = tf.Variable(tf.random_normal([4096,n_classes], stddev=0.01), name='weight')
        classb = tf.Variable(tf.zeros([n_classes]), name='biases')
        classifier = tf.nn.xw_plus_b(fc1_fus, classW, classb, name=scope)

    # prob
    ## softmax(name='prob')
    prob = tf.nn.softmax(classifier, name='prob')
    return prob


def loss(prob, labels, rgb_model, dep_model, tag='fus'):
    def _get_partial_regularizer(scope_name, w_shape):
        #b_shape = [w_shape[-1]]
        with tf.variable_scope(scope_name):
            w = tf.get_variable('weight', w_shape, dtype=tf.float32)
            #b = tf.get_variable('biases', b_shape, dtype=tf.float32)

        # compute l2 loss
        w_l2 = tf.nn.l2_loss(w)
        #b_l2 = tf.nn.l2_loss(b)

        return w_l2 #+ b_l2

    tag += '_'
    logits = tf.log(tf.clip_by_value(prob, 1e-10, 1.0), name='logits')
    L = -tf.reduce_sum(labels * logits, reduction_indices=1)
    loss = tf.reduce_sum(L, reduction_indices=0, name='loss')

    # regularize weights
    '''
    regularizers = _get_partial_regularizer('rgb_fc7', [4096,4096], [4096]) + \
        _get_partial_regularizer('dep_fc7', [4096,4096], [4096]) + \
        _get_partial_regularizer('fus_fc1_fus', [4096*2,4096], [4096]) + \
        _get_partial_regularizer('fus_class', [4096,FLAGS.n_classes], [FLAGS.n_classes])
    '''
    regularizers1, regularizers2 = 0,0
    layers1 = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7']
    for layer in layers1:
        regularizers1 += np.linalg.norm(rgb_model['rgb_'+layer+'W']) + \
                np.linalg.norm(dep_model['dep_'+layer+'W'])
    regularizers2 = _get_partial_regularizer('fus_fc1_fus',[4096*2,4096]) + \
            _get_partial_regularizer('fus_class', [4096,FLAGS.n_classes])
    loss += 1e-4 * (regularizers1 + regularizers2)
    return loss


'''
if __name__== '__main__':
    rgb_model = np.load(os.path.join(cfg.DIR_MODEL, 'rgb_model.npy')).item()
    dep_model = np.load(os.path.join(cfg.DIR_MODEL, 'dep_model.npy')).item()
    rgb_img = tf.Variable(np.random.random((400,227,227,3)), dtype=tf.float32, name='rgb_img')
    dep_img = tf.Variable(np.random.random((400,227,227,3)), dtype=tf.float32, name='dep_img')
    keep_prob = tf.Variable(1.0)
    inference(rgb_img, dep_img, rgb_model, dep_model, keep_prob, tag='fus')
'''
