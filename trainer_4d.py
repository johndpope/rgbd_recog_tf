import tensorflow as tf
import numpy as np
import model_4d as model
import os, sys, time, ipdb
import configure as cfg
from utils import common


# model parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_iter', 500, """Maximum number of training iteration.""")
tf.app.flags.DEFINE_integer('batch_size', 400, """Numer of images to process in a batch.""")
tf.app.flags.DEFINE_integer('img_s', cfg.IMG_S, """"Size of a square image.""")
tf.app.flags.DEFINE_integer('n_classes', 51, """Number of classes.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """"Learning rate for training models.""")
tf.app.flags.DEFINE_integer('summary_frequency', 1, """How often to write summary.""")
tf.app.flags.DEFINE_integer('checkpoint_frequency', 3, """How often to evaluate and write checkpoint.""")


#=========================================================================================================
def placeholder_inputs(batch_size):
    rgbd_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.img_s, FLAGS.img_s, 4), name='rgbd_placeholder')
    labels_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.n_classes), name='labels_placeholder')
    keep_prob_ph = tf.placeholder(tf.float32, shape=(), name='keep_prob_placeholder')
    return rgbd_ph, labels_ph, keep_prob_ph


def fill_feed_dict(rgbd_batch, lbl_batch, rgbd_ph, labels_ph, keep_prob_ph, is_training):
    if lbl_batch.shape[0] < FLAGS.batch_size: # pad the remainder with zeros
        M = FLAGS.batch_size - lbl_batch.shape[0]
        rgbd_batch = np.pad(rgbd_batch, ((0,M),(0,0),(0,0),(0,0)), 'constant', constant_values=0)
        lbl_batch  = np.pad(lbl_batch, ((0,M),(0,0)), 'constant', constant_values=0)

    if is_training:
        feed_dict = {rgbd_ph:rgbd_batch, labels_ph:lbl_batch, keep_prob_ph:0.5}
    else:
        feed_dict = {rgbd_ph:rgbd_batch, labels_ph:lbl_batch, keep_prob_ph:1.0}

    return feed_dict


def do_eval(sess, eval_correct, rgbd_ph, labels_ph, keep_prob_ph, all_data, all_labels, logfile=None):
    true_count, start_idx = 0, 0
    num_samples = all_data.shape[0]

    while start_idx != num_samples:
        stop_idx = common.next_batch(indices, start_idx, FLAGS.batch_size)
        batch_idx = indices[start_idx:stop_idx]

        fd = fill_feed_dict(
                all_data[batch_idx], all_labels[batch_idx],
                rgbd_ph, labels_ph, keep_prob_ph,
                is_training=False)
        true_count += true_count*1.0 / num_samples
        print '    Num-samples:%d  Num-correct:%d  Precision:%0.04f' % (num_samples, true_count, precision)
        if logfile is not None:
            logfile.write('    Num-samples:%d  Num-correct:%d  Precision:%0.04f' % (num_samples, true_count, precision))
    return precision


#=========================================================================================================
def run_training(tag):
    logfile = open(os.path.join(cfg.DIR_LOG, 'training_'+tag+'.log'), 'w', 0)

    # load data
    print 'Loading lists...'
    with open(cfg.PTH_TRAIN_LST, 'r') as f: train_lst = f.read().splitlines()
    with open(cfg.PTH_EVAL_LST,  'r') as f: eval_lst  = f.read().splitlines()
    #train_lst = train_lst[:10]; eval_lst = eval_lst[:10] #TODO

    print 'Loading training data...'
    train_data, train_labels = common.load_4d(train_lst, cfg.DIR_DATA, cfg.DIR_DATA_RAW) 
    print 'Loading validation data...'
    eval_data,  eval_labels  = common.load_4d(eval_lst,  cfg.DIR_DATA, cfg.DIR_DATA_RAW)
    num_train = train_data.shape[0]

    # tensorflow variables and operations
    print 'Preparing tensorflow...'
    rgbd_ph, labels_ph, keep_prob_ph = placeholder_inputs(FLAGS.batch_size)

    prob = model.inference(rgbd_ph, keep_prob_ph, tag)
    loss = model.loss(prob, labels_ph, tag)
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
        # training phase-------------------------------------------------
        start_time = time.time()


        # shuffle indices
        indices = np.random.permutation(num_train)

        # train by batches
        total_loss, start_idx = 0, 0
        ipdb.set_trace()
        while start_idx != num_train:
            stop_idx = common.next_batch(indices, start_idx, FLAGS.batch_size)
            batch_idx = indices[start_idx: stop_idx]

            fd = fill_feed_dict(
                    train_data[batch_idx], train_labels[batch_idx],
                    rgbd_ph, labels_ph, keep_prob_ph, is_training=True)
            _, loss_value = sess.run([train_op, loss], feed_dict=fd)
            assert not np.isnan(loss_value), 'Loss value is NaN'

            total_loss += loss_value
            start_idx = stop_idx

        duration = time.time() - start_time


        # write summary-------------------------------------------------
        if step % FLAGS.summary_frequency == 0:
            print 'Step %d: loss = %.3f (%.3f sec)' % (step, total_loss, duration)
            logfile.write('Step %d: loss = %.3f (%.3f sec)' % (step, total_loss, duration))
            summary_str = sess.run(summary, feed_dict=fd)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
        else:
            print 'Step', step, ' '
            logfile.write('Step %d ' % step)


        # write checkpoint----------------------------------------------
        if step % FLAGS.checkpoint_frequency == 0 or (step+1) == FLAGS.max_iter:
            checkpoint_file = os.path.join(cfg.DIR_CKPT, tag)
            saver.save(sess, checkpoint_file, global_step=step)

            print '  Training data eval:'
            logfile.write('  Training data eval:')
            do_eval(
                    sess, eval_correct,
                    rgbd_ph, labels_ph, keep_prob_ph,
                    train_data, train_labels)
            
            print '  Validation data eval:'
            logfile.write('    Validation data eval:')
            precision = do_eval(
                    sess, eval_correct,
                    images_ph, labels_ph, keep_prob_ph,
                    eval_data, eval_labels)

            # early stopping
            to_stop, patience_count = common.early_stopping(old_precision, precision, patience_count)
            old_precision = precision
            if to_stop:
                print 'Early stopping...'
                logfile.write('Early stopping...')
                break
    logfile.close()
    return


#=======================================================================
def main(argv=None):
    with tf.Graph().as_default():
        run_training(tag='4d')


if __name__ == '__main__':
    tf.app.run(main)
