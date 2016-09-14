import tensorflow as tf
import numpy as np
import single_channel_model as model
import os, sys, time, ipdb
from utils import common
import configure as cfg 


# model parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_iter', 10000, """Maximum number of training iteration.""")
tf.app.flags.DEFINE_integer('batch_size', 400, """Numer of images to process in a batch.""")
tf.app.flags.DEFINE_integer('img_s', cfg.IMG_S, """"Size of a square image.""")
tf.app.flags.DEFINE_integer('n_classes', 51, """Number of classes.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """"Learning rate for training models.""")
tf.app.flags.DEFINE_integer('summary_frequency', 1, """How often to write summary.""")
tf.app.flags.DEFINE_integer('checkpoint_frequency', 5, """How often to evaluate and write checkpoint.""")


#=========================================================================================
def placeholder_inputs(batch_size):
    images_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.img_s, FLAGS.img_s, 3), 
            name='images_placeholder') 
    labels_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.n_classes),
            name='labels_placeholder')
    keep_prob_ph = tf.placeholder(tf.float32, shape=(), name='keep_prob_placeholder')

    return images_ph, labels_ph, keep_prob_ph


def fill_feed_dict(lst, images_ph, labels_ph, keep_prob_ph, tag, is_training):
    """Fills the feed_dict for training the given step
    """
    popped = lst[:FLAGS.batch_size]
    lst[:FLAGS.batch_size] = []

    if tag == 'rgb':
        img, lbl = common.load_images(popped, cfg.DIR_DATA, cfg.EXT_RGB, cfg.CLASSES)
    if tag == 'dep':
        img, lbl = common.load_images(popped, cfg.DIR_DATA, cfg.EXT_D, cfg.CLASSES)

    if img.shape[0] < FLAGS.batch_size: # pad the remainder with zeros
        N = FLAGS.batch_size - img.shape[0]
        img = np.pad(img, ((0,N),(0,0),(0,0),(0,0)), 'constant', constant_values=0)
        lbl = np.pad(lbl, ((0,N),(0,0)), 'constant', constant_values=0)

    if is_training:
        feed_dict = {images_ph: img, labels_ph: lbl, keep_prob_ph: 0.5}
    else:
        feed_dict = {images_ph: img, labels_ph: lbl, keep_prob_ph: 1.0}
    return feed_dict, lst


def do_eval(sess, logits, eval_correct, images_ph, labels_ph, keep_prob_ph, data_list, tag):
    true_count = 0
    num_samples = len(data_list)

    using_list = data_list[:]
    while using_list != []:
        fd, using_list = fill_feed_dict(using_list, images_ph, labels_ph, keep_prob_ph, tag, is_training=False)
        true_count += sess.run(eval_correct, feed_dict=fd)

    precision = true_count*1.0 / num_samples
    print '  Num samples:%d Num correct:%d Precision:%0.04f' % (num_samples, true_count, precision)
    return precision


#=========================================================================================
def run_training(tag):
    # load data
    print 'Loading data...'
    net_data = np.load(cfg.PTH_WEIGHT_ALEX).item()
    with open(cfg.PTH_TRAIN_LST, 'r') as f: train_lst = f.read().splitlines()
    with open(cfg.PTH_EVAL_LST, 'r') as f: eval_lst = f.read().splitlines()


    # tensorflow variables and operations
    print 'Preparing tensorflow...'
    images_ph, labels_ph, keep_prob_ph = placeholder_inputs(FLAGS.batch_size)

    logits = model.inference(images_ph, net_data, keep_prob_ph)
    loss = model.loss(logits, labels_ph)
    train_op = model.training(loss)
    eval_correct = model.evaluation(logits, labels_ph)
    init_op = tf.initialize_all_variables()

    # tensorflow monitor
    summary = tf.merge_all_summaries()
    saver = tf.train.Saver()
   
    # initialize graph
    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter(cfg.DIR_SUMMARY, sess.graph)
    sess.run(init_op)


    # start the training loop
    old_precision = sys.maxsize
    patience_count = 0
    print 'Start the training loop...'
    for step in range(FLAGS.max_iter):
        # training phase----------------------------------------------
        start_time = time.time()
        using_lst = train_lst[:] #clone train_lst
        np.random.shuffle(using_lst)

        total_loss = 0
        while using_lst != []:
            fd, using_lst = fill_feed_dict(using_lst, images_ph, labels_ph, keep_prob_ph, tag, is_training=True)
            _, loss_value = sess.run([train_op, loss], feed_dict=fd)
            assert not np.isnan(loss_value), 'Loss value is NaN'
            total_loss += loss_value

        duration = time.time() - start_time


        # write summary------------------------------------------------
        if step % FLAGS.summary_frequency == 0:
            print 'Step %d: loss = %.3f (%.3f sec)' % (step, total_loss, duration)
            summary_str = sess.run(summary, feed_dict=fd)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
        else:
            print 'Step', step, '  '


        # write checkpoint---------------------------------------------
        if step % FLAGS.checkpoint_frequency == 0 or (step+1) == FLAGS.max_iter:
            checkpoint_file = os.path.join(cfg.DIR_CKPT, tag)
            saver.save(sess, checkpoint_file, global_step=step)

            print ' Training data eval:'
            do_eval(sess, logits, eval_correct, images_ph, labels_ph, keep_prob_ph, train_lst, tag)

            print ' Validation data eval:'
            precision = do_eval(sess, logits, eval_correct, images_ph, labels_ph, keep_prob_ph, eval_lst, tag)

            # early stopping
            to_stop, patience_count = common.early_stopping(old_precision, precision, patience_count)
            old_precision = precision
            if to_stop: 
                print 'Early stopping...'
                break
    return


#=========================================================================================
def main(argv=None):
    with tf.Graph().as_default():
        run_training(tag='rgb')
        run_training(tag='dep')


if __name__ == '__main__':
    tf.app.run(main)
