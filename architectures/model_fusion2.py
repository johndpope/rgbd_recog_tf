import os, ipdb
import tensorflow as tf
import numpy as np
import configure as cfg
from tfcommon import *

FLAGS = tf.app.flags.FLAGS


def inference(rgb_feat, dep_feat, keep_prob, tag='fus'):
    tag += '_'
    batch_size = FLAGS.batch_size
    n_classes = FLAGS.n_classes

    concat_feat = tf.concat(1, [rgb_feat, dep_feat], name='concat_feat')

    # fc1-fus
    with tf.name_scope(tag+'fc1_fus') as scope:
        fc1_fusW = tf.Variable(tf.random_normal([4096*2,4096], stddev=0.01), name='weight')
        fc1_fusb = tf.Variable(tf.zeros([4096]), name='biases')
        fc1_fus  = tf.nn.relu_layer(concat_feat, fc1_fusW, fc1_fusb, name=scope)

    # classifier
    with tf.name_scope(tag+'class') as scope:
        classW = tf.Variable(tf.random_normal([4096,n_classes], stddev=0.01), name='weight')
        classb = tf.Variable(tf.zeros([n_classes]), name='biases')
        classifier = tf.nn.xw_plus_b(fc1_fus, classW, classb, name=scope)

    # prob
    #prob = tf.nn.softmax(classifier, name='prob')
    return classifier #prob


def loss(score, labels, tag='fus'):
    def _get_partial_regularizer(scope_name, w_shape):
        with tf.variable_scope(scope_name):
            w = tf.get_variable('weight', w_shape, dtype=tf.float32)
            #b = tf.get_variable('biases', w_shape[-1], dtype=tf.float32)

        # compute l2 loss
        w_l2 = tf.nn.l2_loss(w)
        #b_l2 = tf.nn.l2_loss(b)

        return w_l2 #+ b_l2

    '''
    tag += '_'
    logits = tf.log(tf.clip_by_value(prob, 1e-10, 1.0), name='logits')
    L = -tf.reduce_sum(labels * logits, reduction_indices=1)
    loss = tf.reduce_sum(L, reduction_indices=0, name='loss')
    '''

    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(score, labels), name='loss')

    # regularize weights
    regularizers1, regularizers2 = 0,0

    '''
    layers1 = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7']
    for layer in layers1:
        regularizers1 += np.linalg.norm(rgb_model['rgb_'+layer+'W']) + \
                np.linalg.norm(dep_model['dep_'+layer+'W'])
    '''
    regularizers2 = _get_partial_regularizer('fus_fc1_fus',[4096*2,4096]) + \
            _get_partial_regularizer('fus_class', [4096,FLAGS.n_classes])
    #loss += 1e-4 * (regularizers1 + regularizers2)
    loss += 1e-4 * regularizers2
    return loss

