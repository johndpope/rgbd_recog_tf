import tensorflow as tf
import numpy as np
import os, sys, time, ipdb
from utils import common
import configure as cfg 
from architectures import model_single_channel as model


# model parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_iter', 30000, """Maximum number of training iteration.""")
tf.app.flags.DEFINE_integer('batch_size', 400, """Numer of images to process in a batch.""")
tf.app.flags.DEFINE_integer('img_s', cfg.IMG_S, """"Size of a square image.""")
tf.app.flags.DEFINE_integer('img_s_raw', 256, """"Size of a square image.""")
tf.app.flags.DEFINE_integer('n_classes', 51, """Number of classes.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """"Learning rate for training models.""")
tf.app.flags.DEFINE_integer('summary_frequency', 1, """How often to write summary.""")
tf.app.flags.DEFINE_integer('checkpoint_frequency', 3, """How often to evaluate and write checkpoint.""")


#=========================================================================================
def placeholder_inputs(batch_size):
    """Create placeholders for tensorflow with some specific batch_size

    Args:
        batch_size: size of each batch

    Returns:
        images_ph: 4D tensor of shape [batch_size, image_size, image_size, 3]
        labels_ph: 2D tensor of shape [batch_size, num_classes]
        keep_prob_ph: 1D tensor for the keep_probability (for dropout during training)
    """
    images_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.img_s, FLAGS.img_s, 3), name='images_placeholder') 
    labels_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.n_classes), name='labels_placeholder')
    keep_prob_ph = tf.placeholder(tf.float32, shape=(), name='keep_prob_placeholder')

    return images_ph, labels_ph, keep_prob_ph


def fill_feed_dict(img_batch, lbl_batch, images_ph, labels_ph, keep_prob_ph, is_training):
    """Fills the feed_dict. If the batch has fewer samples than the placeholder, it is padded
    with zeros.

    Args:
        img_batch: 4D numpy array of shape [batch_size, image_size, image_size, 3]
        lbl_batch: 2D numpy array of shape [batch_size, num_classes]
        images_ph: 4D tensor of shape [batch_size, image_size, image_size, 3]
        labels_ph: 2D tensor of shape [batch_size, num_classes]
        keep_prob_ph: 1D tensor for the keep_probability (for dropout during training)
        is_training: True or False. If True, keep_prob = 0.5, 1.0 otherwise

    Returns:
        feed_dict: feed dictionary
    """
    if img_batch.shape[0] < FLAGS.batch_size: # pad the remainder with zeros
        M = FLAGS.batch_size - img_batch.shape[0]
        img_batch = np.pad(img_batch, ((0,M),(0,0),(0,0),(0,0)), 'constant', constant_values=0)
        lbl_batch = np.pad(lbl_batch, ((0,M),(0,0)), 'constant', constant_values=0)

    if is_training:
        feed_dict = {images_ph: common.random_crop(img_batch, True), labels_ph: lbl_batch, keep_prob_ph: 0.5}
    else:
        feed_dict = {images_ph: common.central_crop(img_batch), labels_ph: lbl_batch, keep_prob_ph: 1.0}
    return feed_dict


def do_eval(sess, eval_correct, images_ph, labels_ph, keep_prob_ph, all_data, all_labels, logfile=None):
    ''' Run one evaluation against the full epoch of data

    Args:
        sess: the session in which the model has been trained
        eval_correct: the tensor that returns the number of correct predictions
        images_ph: tensor place holder for images
        labels_ph: tensor place holder for labels
        keep_prob_ph: tensor place holder for keep_prob
        all_data: all loaded images
        all_labels: all labels corresponding to all_images

    Return
        precision: percentage of correct recognition
    '''
    true_count, start_idx = 0, 0
    num_samples = all_data.shape[0]
    indices = np.arange(num_samples)
    while start_idx != num_samples:
        stop_idx = common.next_batch(indices, start_idx, FLAGS.batch_size)
        batch_idx = indices[start_idx: stop_idx]

        fd = fill_feed_dict(
            all_data[batch_idx], all_labels[batch_idx], 
            images_ph, labels_ph, keep_prob_ph, 
            is_training=False)
        true_count += sess.run(eval_correct, feed_dict=fd)
        start_idx = stop_idx

    precision = true_count*1.0 / num_samples
    print '    Num-samples:%d   Num-correct:%d   Precision:%0.04f' % (num_samples, true_count, precision)
    if logfile is not None:
        logfile.write('    Num-samples:%d  Num-correct:%d  Precision:%0.04f\n' % (num_samples, true_count, precision))

    return precision


#=========================================================================================
def run_training(tag):
    logfile = open(os.path.join(cfg.DIR_LOG, 'training_'+tag+'.log'), 'w', 0)

    # load data
    print 'Loading AlexNet...'
    net_data = np.load(cfg.PTH_WEIGHT_ALEX).item()

    print 'Loading lists...'
    with open(cfg.PTH_TRAIN_LST, 'r') as f: train_lst = f.read().splitlines()
    eval_lst = []
    for trial in range(cfg.N_TRIALS):
        with open(cfg.PTH_EVAL_LST[trial], 'r') as f: eval_lst.append(f.read().splitlines())
    if tag == 'rgb': ext = cfg.EXT_RGB
    elif tag == 'dep': ext = cfg.EXT_D
    #train_lst = train_lst[:10] #TODO
    #cfg.N_TRIALS = 1 #TODO

    print 'Loading training data...'
    train_data, train_labels = common.load_images(train_lst, cfg.DIR_DATA, ext, cfg.CLASSES)
    num_train = len(train_data)

    print 'Loading validation data...'
    eval_data, eval_labels = [], []
    for trial in range(cfg.N_TRIALS):
        print 'Trial %d:' % (trial+1)
        #pth = cfg.DIR_DATA if tag == 'rgb' else cfg.DIR_DATA_EVAL # TODO: use original rgb images
        #foo, bar = common.load_images(eval_lst[trial], pth, ext, cfg.CLASSES) # TODO
        foo, bar = common.load_images(eval_lst[trial], cfg.DIR_DATA_EVAL, ext, cfg.CLASSES)
        eval_data.append(foo)
        eval_labels.append(bar)

    # tensorflow variables and operations
    print 'Preparing tensorflow...'
    images_ph, labels_ph, keep_prob_ph = placeholder_inputs(FLAGS.batch_size)

    prob = model.inference(images_ph, net_data, keep_prob_ph, tag)
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
        # training phase----------------------------------------------
        start_time = time.time()

        # shuffle indices
        indices = np.random.permutation(num_train)

        # train by batches
        total_loss, start_idx = 0, 0
        lim = 10
        while start_idx != num_train:
            if start_idx*100.0 / num_train > lim:
                print 'Trained %d/%d' % (start_idx, num_train)
                lim += 10

            stop_idx = common.next_batch(indices, start_idx, FLAGS.batch_size)
            batch_idx = indices[start_idx: stop_idx]

            fd = fill_feed_dict(
                train_data[batch_idx], train_labels[batch_idx], 
                images_ph, labels_ph, keep_prob_ph,
                is_training=True)
            _, loss_value = sess.run([train_op, loss], feed_dict=fd)
            assert not np.isnan(loss_value), 'Loss value is NaN'

            total_loss += loss_value
            start_idx = stop_idx

        duration = time.time() - start_time


        # write summary------------------------------------------------
        if step % FLAGS.summary_frequency == 0:
            print 'Step %d: loss = %.3f (%.3f sec)' % (step, total_loss, duration)
            logfile.write('Step %d: loss = %.3f (%.3f sec)\n' % (step, total_loss, duration))
            summary_str = sess.run(summary, feed_dict=fd)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
        else:
            print 'Step', step, '  '
            logfile.write('Step %d \n' % step)


        # write checkpoint---------------------------------------------
        if step % FLAGS.checkpoint_frequency == 0 or (step+1) == FLAGS.max_iter:
            checkpoint_file = os.path.join(cfg.DIR_CKPT, tag)
            saver.save(sess, checkpoint_file, global_step=step)
            
            print '  Training data eval:'
            logfile.write('  Training data eval:\n')
            do_eval(
                sess, eval_correct, 
                images_ph, labels_ph, keep_prob_ph, 
                train_data, train_labels, logfile)

            print '  Validation data eval:'
            logfile.write('  Validation data eval:\n')
            avg_precision = 0
            for trial in range(cfg.N_TRIALS):
                precision = do_eval(
                    sess, eval_correct, 
                    images_ph, labels_ph, keep_prob_ph, 
                    eval_data[trial], eval_labels[trial], logfile)
                avg_precision += precision
            avg_precision /= cfg.N_TRIALS
            print 'Average precision: %.4f' % avg_precision
            logfile.write('Average precision: %.4f\n' % avg_precision)

            # early stopping
            to_stop, patience_count = common.early_stopping(old_precision, avg_precision, patience_count, tolerance=1e-3, patience_limit=10)
            old_precision = avg_precision
            if to_stop: 
                print 'Early stopping...'
                logfile.write('Early stopping...\n')
                break
    logfile.close()
    return


#=========================================================================================
def main(argv=None):
    with tf.Graph().as_default():
        run_training(tag='rgb')
    with tf.Graph().as_default():
        run_training(tag='dep')


if __name__ == '__main__':
    tf.app.run(main)
