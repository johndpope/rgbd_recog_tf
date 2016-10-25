import tensorflow as tf
import numpy as np
import model_fusion as model
import os, sys, time, ipdb
import configure as cfg
from utils import common


# model parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_iter', 500, """Maximum number of training iteration.""")
tf.app.flags.DEFINE_integer('batch_size', 200, """Numer of images to process in a batch.""")
tf.app.flags.DEFINE_integer('img_s', cfg.IMG_S, """"Size of a square image.""")
tf.app.flags.DEFINE_integer('n_classes', 51, """Number of classes.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """"Learning rate for training models.""")
tf.app.flags.DEFINE_integer('summary_frequency', 1, """How often to write summary.""")
tf.app.flags.DEFINE_integer('checkpoint_frequency', 3, """How often to evaluate and write checkpoint.""")

#==================================================================================================
def placeholder_inputs(batch_size):
    """Create placeholders for tensorflow with some specific batch_size

    Args:
        batch_size: size of each batch

    Returns:
        rgb_ph: 4D tensor of shape [batch_size, image_size, image_size, 3]
        dep_ph: 4D tensor of shape [batch_size, image_size, image_size, 3]
        labels_ph: 2D tensor of shape [batch_size, num_classes]
        keep_prob_ph: 1D tensor for the keep_probability (for dropout during training)
    """
    rgb_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.img_s, FLAGS.img_s, 3), name='rgb_placeholder')
    dep_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.img_s, FLAGS.img_s, 3), name='dep_placeholder')
    labels_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.n_classes), name='labels_placeholder')
    keep_prob_ph = tf.placeholder(tf.float32, shape=(), name='keep_prob_placeholder')
    return rgb_ph, dep_ph, labels_ph, keep_prob_ph


def fill_feed_dict(rgb_batch, dep_batch, lbl_batch, rgb_ph, dep_ph, labels_ph, keep_prob_ph, is_training):
    """Fills the feed_dict. If the batch has fewer samples than the placeholder, it is padded
    with zeros.

    Args:
        rgb_batch: 4D numpy array of shape [batch_size, image_size, image_size, 3]
        dep_batch: 4D numpy array of shape [batch_size, image_size, image_size, 3]
        lbl_batch: 2D numpy array of shape [batch_size, num_classes]
        rgb_ph: tensor place holder for rgb_batch
        dep_ph: tensor place holder for dep_batch
        labels_ph: 2D tensor of shape [batch_size, num_classes]
        keep_prob_ph: 1D tensor for the keep_probability (for dropout during training)
        is_training: True or False. If True, keep_prob = 0.5, 1.0 otherwise

    Returns:
        feed_dict: feed dictionary
    """
    if lbl_batch.shape[0] < FLAGS.batch_size: # zero-padding
        M = FLAGS.batch_size - lbl_batch.shape[0]
        rgb_batch = np.pad(rgb_batch, ((0,M),(0,0),(0,0),(0,0)), 'constant', constant_values=0)
        dep_batch = np.pad(dep_batch, ((0,M),(0,0),(0,0),(0,0)), 'constant', constant_values=0)
        lbl_batch = np.pad(lbl_batch, ((0,M),(0,0)), 'constant', constant_values=0)

    if is_training:
        feed_dict = {rgb_ph:rgb_batch, dep_ph:dep_batch, labels_ph:lbl_batch, keep_prob_ph: 0.5}
    else:
        feed_dict = {rgb_ph:rgb_batch, dep_ph:dep_batch, labels_ph:lbl_batch, keep_prob_ph: 1.0}
    return feed_dict


def do_eval(sess, eval_correct, rgb_ph, dep_ph, labels_ph, keep_prob_ph, all_rgb, all_dep, all_labels):
    ''' Run one evaluation against the full epoch of data

    Args:
        sess: the session in which the model has been trained
        eval_correct: the tensor that returns the number of correct predictions
        rgb_ph: tensor place holder for color images
        dep_ph: tensor place holder for depth images
        labels_ph: tensor place holder for labels
        keep_prob_ph: tensor place holder for keep_prob
        all_rgb: all loaded color images
        all_dep: all loaded depth images
        all_labels: all labels corresponding to all_images

    Return
        precision: percentage of correct recognition
    '''
    true_count, start_idx = 0, 0
    num_samples = all_labels.shape[0]
    #indices = np.random.permutation(num_samples)
    indices = np.arange(num_samples)
    while start_idx != num_samples:
        stop_idx = common.next_batch(indices, start_idx, FLAGS.batch_size)
        batch_idx = indices[start_idx: stop_idx]
        
        fd = fill_feed_dict(
                all_rgb[batch_idx], all_dep[batch_idx], all_labels[batch_idx],
                rgb_ph, dep_ph, labels_ph, keep_prob_ph,
                is_training=False)
        true_count += sess.run(eval_correct, feed_dict=fd)
        start_idx = stop_idx
        
    precision = true_count*1.0 / num_samples
    print '    Num-samples:%d   Num-correct:%d   Precision:%0.04f' % (num_samples, true_count, precision)
    return precision


#==================================================================================================
def run_training(tag='fus'):
    # load data-----------------------------------------------------------------------------------
    print 'Loading RGB and depth models...'
    rgb_model = np.load(cfg.PTH_RGB_MODEL).item()
    dep_model = np.load(cfg.PTH_DEP_MODEL).item()
    print 'Loading lists...'
    with open(cfg.PTH_TRAIN_LST, 'r') as f: train_lst = f.read().splitlines()
    with open(cfg.PTH_EVAL_LST,  'r') as f: eval_lst  = f.read().splitlines()
    print 'Loading training data...'
    rgb_train_data, train_labels = common.load_images(train_lst, cfg.DIR_DATA, cfg.EXT_RGB, cfg.CLASSES)
    dep_train_data, _            = common.load_images(train_lst, cfg.DIR_DATA, cfg.EXT_D, cfg.CLASSES)
    print 'Loading validation data...'
    rgb_eval_data,  eval_labels  = common.load_images(eval_lst,  cfg.DIR_DATA, cfg.EXT_RGB, cfg.CLASSES)
    dep_eval_data,  _            = common.load_images(eval_lst,  cfg.DIR_DATA, cfg.EXT_D, cfg.CLASSES)
    num_train = rgb_train_data.shape[0]

    # tensorflow variables and operations----------------------------------------------------------
    print 'Preparing tensorflow...'
    rgb_ph, dep_ph, labels_ph, keep_prob_ph = placeholder_inputs(FLAGS.batch_size)

    prob = model.inference(rgb_ph, dep_ph, rgb_model, dep_model, keep_prob_ph)
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
    old_precision = sys.maxsize
    patience_count = 0
    print 'Start the training loop...'
    for step in range(FLAGS.max_iter):
        # training phase----------------------------------------------------------------------------
        start_time = time.time()

        # shuffle indices
        indices = np.random.permutation(num_train)

        # train by batches
        total_loss, start_idx = 0, 0
        while start_idx != num_train:
            stop_idx = common.next_batch(indices, start_idx, FLAGS.batch_size)
            batch_idx = indices[start_idx: stop_idx]

            fd = fill_feed_dict(
                    rgb_train_data[batch_idx], dep_train_data[batch_idx], train_labels[batch_idx],
                    rgb_ph, dep_ph, labels_ph, keep_prob_ph, is_training=True)
            _, loss_value = sess.run([train_op, loss], feed_dict=fd)
            assert not np.isnan(loss_value), 'Loss value is NaN'

            total_loss += loss_value
            start_idx = stop_idx

        duration = time.time() - start_time

        # write summary----------------------------------------------------------------------------
        if step % FLAGS.summary_frequency == 0:
            print 'Step %d: loss = %.3f (%.3f sec)' % (step, total_loss, duration)
            summary_str = sess.run(summary, feed_dict=fd)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
        else:
            print 'Step', step, ' '


        # write checkpoint-------------------------------------------------------------------------
        if step % FLAGS.checkpoint_frequency == 0 or (step+1) == FLAGS.max_iter:
            checkpoint_file = os.path.join(cfg.DIR_CKPT, tag)
            saver.save(sess, checkpoint_file, global_step=step)

            print '  Training data eval:'
            do_eval(
                    sess, eval_correct, rgb_ph, dep_ph, labels_ph, keep_prob_ph, 
                    rgb_train_data, dep_train_data, train_labels)

            print '  Validation data eval:'
            precision = do_eval(
                    sess, eval_correct, rgb_ph, dep_ph, labels_ph, keep_prob_ph,
                    rgb_eval_data, dep_eval_data, eval_labels)

            # early stopping
            to_stop, patience_count = common.early_stopping(old_precision, precision, patience_count)
            old_precision = precision
            if to_stop:
                print 'Early stopping...'
                break
    return


#==================================================================================================
def main(argv=None):
    with tf.Graph().as_default():
        run_training()
    return


if __name__ == '__main__':
    tf.app.run(main)
