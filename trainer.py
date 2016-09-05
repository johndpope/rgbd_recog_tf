import tensorflow as tf
import numpy as np
import single_channel_model as model1
import os, sys, ipdb
from utils import common


DIR_CKPT    = 'checkpoints'
DIR_LST     = 'lists'
DIR_MODEL   = 'models'
DIR_SUMMARY = 'summary'
DIR_DATA    = os.path.join('/','home','knmac','washington','fullset','cropped')


LST_TRAIN   = os.path.join(DIR_LST, 'train_full.lst')
LST_EVAL    = os.path.join(DIR_LST, 'eval_full.lst')
LST_DICT    = os.path.join(DIR_LST, 'dictionary_full.lst')
WEIGHT_ALEX = os.path.join(DIR_MODEL, 'bvlc_alexnet.npy')

CLASSES = open(LST_DICT, 'r').read().splitlines()

# basic model parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_iter', 1000, """Maximum number of training iteration.""")
tf.app.flags.DEFINE_integer('batch_size', 200, """Numer of images to process in a batch.""")
tf.app.flags.DEFINE_integer('img_s', 227, """"Size of a square image.""")
tf.app.flags.DEFINE_integer('n_classes', 51, """Number of classes.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-4, """"Learning rate for training models.""")


#=========================================================================================
def placeholder_inputs(batch_size=None):
    images_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.img_s, FLAGS.img_s, 3)) #TODO:check dimension's order
    labels_ph = tf.placeholder(tf.int32, shape=(batch_size, FLAGS.n_classes))

    return images_ph, labels_ph


def fill_feed_dict(train_idx, batch_idx):
    """Fills the feed_dict for training the given step
    """
    N = train_idx.shape[0]

    if batch_id + FLAGS.batch_size > N:
        feed_dict = {}
        batch_idx = -1
    else:
        feed_dict = {}
        batch_idx += FLAGS.batch_size
    return feed_dict, batch_idx


#=========================================================================================
def run_training():
    net_data = np.load(WEIGHT_ALEX).item()

    # tensorflow variables and operations
    images_ph, labels_ph = placeholder_inputs()

    prob = model1.inference(images_ph, net_data, is_training=True)
    loss = model1.loss(prob, labels_ph)
    train_op = model1.training(loss)
    eval_correct = model1.evaluation(prob, labels_ph)
    init_op = tf.initialize_all_variables()

    # tensorflow monitor
    summary = tf.merge_all_summaries()
    saver = tf.train.Saver()
   
    # initialize graph
    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter(DIR_SUMMARY, sess.graph)
    sess.run(init_op)


    # start the training loop
    for step in range(FLAGS.max_iter):
        # training phase
        rgb_paths, dep_paths, labels = common.get_paths_labels(DIR_DATA, LST_TRAIN, CLASSES, to_shuffle=True)

        batch_idx = 0
        while batch_idx != -1:
            fd, batch_idx = fill_feed_dict(train_idx, batch_idx)

        # evaluation phase
        rgb_paths, dep_paths, labels = common.get_paths_labels(DIR_DATA, LST_EVAL, CLASSES, to_shuffle=False)
    return


#=========================================================================================
def main(argv=None):
    with tf.Graph().as_default():
        run_training()


if __name__ == '__main__':
    tf.app.run(main)
