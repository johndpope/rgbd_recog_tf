import numpy as np
import tensorflow as tf
import configure as cfg
import os, ipdb



def main():
    #net_data = np.load(cfg.PTH_WEIGHT_ALEX).item()


    net_data = np.load(cfg.PTH_WEIGHT_ALEX).item()
    with tf.variable_scope('rgb'):
        with tf.name_scope('conv1'):
            conv1_rgb = tf.Variable(tf.zeros(net_data['conv1'][0].shape, name='weight'))
    
    with tf.variable_scope('dep'):
        with tf.name_scope('conv1'):
            conv1_dep = tf.Variable(tf.zeros(net_data['conv1'][0].shape, name='weight'))
    '''
    with tf.variable_scope('rgb'):
        with tf.name_scope('conv1'):
            conv1_rgb = tf.Variable(tf.zeros([11,11,3,96]), name='weight')  # dummy variable !!!
    '''

    saver = tf.train.Saver()
    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)


    ipdb.set_trace()
    saver.restore(sess, 'checkpoints/dep-10')
    return



if __name__ == '__main__':
    with tf.Graph().as_default():
        main()
