import tensorflow as tf
import numpy as np
import single_channel_model as model1
import os, sys, ipdb
from utils import common


DIR_CKPT    = 'checkpoints'
DIR_LST     = 'lists'
DIR_MODEL   = 'models'
DIR_SUMMARY = 'summary'

LST_TRAIN   = os.path.join(DIR_LST, '')
LST_EVAL    = os.path.join(DIR_LST, '')
WEIGHT_ALEX = os.path.join(DIR_MODEL, 'bvlc_alexnet.npy')


# basic model parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 128, """Numer of images to process in a batch.""")
tf.app.flags.DEFINE_integer('img_s', 227, """"Size of a square image.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-4, """"Learning rate for training models.""")

#=========================================================================================
def placeholder_inputs(batch_size=None):
    images_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.img_s, FLAGS.img_s, 3)) #TODO:check dimension's order
    labels_ph = tf.placeholder(tf.int32, shape=(batch_size, 1000))

    return images_ph, labels_ph


def fill_feed_dict():
    """Fills the feed_dict for training the given step
    """
    return


#=========================================================================================
def run_training():
    images_ph, labels_ph = placeholder_inputs()

    net_data = np.load(WEIGHT_ALEX).item()
    prob = model1.inference(images_ph, net_data, is_training=True)
    loss = model1.loss(prob, labels_ph)
    train_op = model1.training(loss)
    eval_correct = model1.evaluation(prob, labels_ph)


    summary = tf.merge_all_summaries()
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()

    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter(DIR_SUMMARY, sess.graph)


    sess.run(init_op)
    return


#=========================================================================================
def main(argv=None):
    with tf.Graph().as_default():
        run_training()


if __name__ == '__main__':
    tf.app.run(main)
