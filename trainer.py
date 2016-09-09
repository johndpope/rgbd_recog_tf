import tensorflow as tf
import numpy as np
import single_channel_model as model
import os, sys, time, ipdb
from utils import common
import configure as cfg 


# basic model parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_iter', 2000, """Maximum number of training iteration.""")
tf.app.flags.DEFINE_integer('batch_size', 100, """Numer of images to process in a batch.""")
tf.app.flags.DEFINE_integer('img_s', cfg.IMG_S, """"Size of a square image.""")
tf.app.flags.DEFINE_integer('n_classes', 51, """Number of classes.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """"Learning rate for training models.""")


#=========================================================================================
def placeholder_inputs(batch_size=None):
    images_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.img_s, FLAGS.img_s, 3), 
            name='images_placeholder') #TODO:check dimension's order
    labels_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.n_classes),
            name='labels_placeholder')

    return images_ph, labels_ph


def fill_feed_dict(lst, batch_idx, images_ph, labels_ph, use_rbg, use_dep):
    """Fills the feed_dict for training the given step
    """
    N = len(lst)

    start = batch_idx
    if batch_idx + FLAGS.batch_size > N:
        stop = N
        batch_idx = -1
    else:
        stop = batch_idx + FLAGS.batch_size
        batch_idx += FLAGS.batch_size

    if use_rbg:
        img, lbl = common.load_images(lst[start:stop], cfg.DIR_DATA, cfg.EXT_RGB, cfg.CLASSES)
    if use_dep:
        img, lbl = common.load_images(lst[start:stop], cfg.DIR_DATA, cfg.EXT_D, cfg.CLASSES)
    feed_dict = {images_ph: img, labels_ph: lbl}
    return feed_dict, batch_idx


def do_eval(sess, eval_correct, images_ph, labels_ph, data_set):
    true_count = 0
    #TODO
    return


#=========================================================================================
def run_training(use_rgb, use_dep):
    # load data
    print 'Loading data...'
    net_data = np.load(cfg.PTH_WEIGHT_ALEX).item()
    with open(cfg.PTH_TRAIN_LST, 'r') as f: train_lst = f.read().splitlines()
    with open(cfg.PTH_EVAL_LST, 'r') as f: eval_lst = f.read().splitlines()


    # tensorflow variables and operations
    print 'Preparing tensorflow...'
    images_ph, labels_ph = placeholder_inputs()

    prob = model.inference(images_ph, net_data, is_training=True)
    loss = model.loss(prob, labels_ph)
    train_op = model.training(loss)
    eval_correct = model.evaluation(prob, labels_ph)
    init_op = tf.initialize_all_variables()

    # tensorflow monitor
    summary = tf.merge_all_summaries()
    saver = tf.train.Saver()
   
    # initialize graph
    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter(cfg.DIR_SUMMARY, sess.graph)
    sess.run(init_op)


    # start the training loop
    print 'Start the training loop...'
    for step in range(FLAGS.max_iter):
        # training phase----------------------------------------------
        start_time = time.time()
        np.random.shuffle(train_lst)

        batch_idx = 0
        while batch_idx != -1:
            fd, batch_idx = fill_feed_dict(train_lst, batch_idx, images_ph, labels_ph, use_rgb, use_dep)
            _, loss_value = sess.run([train_op, loss], feed_dict=fd)

        duration = time.time() - start_time


        # write summary------------------------------------------------
        if step % 100 == 0:
            print 'Step %d: loss = %.3f (%.3f sec)' % (step, loss_value, duration)
            summary_str = sess.run(summary, feed_dict=fd)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()


        # write checkpoint---------------------------------------------
        if (step+1)%100 == 0 or (step+1) == FLAGS.max_iter:
            checkpoint_file = os.path.join(cfg.DIR_CKPT, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=step)
            ipdb.set_trace()
            #TODO:do_eval()
    return


#=========================================================================================
def main(argv=None):
    with tf.Graph().as_default():
        run_training(use_rgb=True, use_dep=False)


if __name__ == '__main__':
    tf.app.run(main)
