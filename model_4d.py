import os, ipdb
import tensorflow as tf
import numpy as np
import configure as cfg
from utils.tfcommon import *

FLAGS = tf.app.flags.FLAGS

def inference(rgbd, keep_prob, tag='4d'):
    tag += '_'

    # conv-1 layer
    with tf.name_scope(tag+'conv1') as scope:
        conv1W = tf.Variable(tf.truncated_normal([11,11,4,96]), name='weight')
        conv1b = tf.Variable(tf.zeros([96]))
        conv1_in = conv(rgbd, conv1W, conv1b, 11, 11, 96, 4, 4, padding='SAME', group=1)
        conv1 = tf.nn.relu(conv1_in, name=scope)
    lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0, name='norm1')
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool1')

    # conv-2 layer

    # conv-3 layer

    # conv-4 layer

    # conv-5 layer

    # fc6 layer

    # fc7 layer

    # fc8 layer
    return
