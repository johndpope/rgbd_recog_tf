import tensorflow as tf
import numpy as np
import bi_channel_model as model
import ipdb
import configure as cfg


# model parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_iter', 1000, """Maximum number of training iteration.""")
tf.app.flags.DEFINE_integer('batch_size', 200, """Numer of images to process in a batch.""")
tf.app.flags.DEFINE_integer('img_s', cfg.IMG_S, """"Size of a square image.""")
tf.app.flags.DEFINE_integer('n_classes', 51, """Number of classes.""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """"Learning rate for training models.""")
tf.app.flags.DEFINE_integer('summary_frequency', 1, """How often to write summary.""")
tf.app.flags.DEFINE_integer('checkpoint_frequency', 5, """How often to evaluate and write checkpoint.""")

#==================================================================================================
def placeholder_inputs(batch_size):
    rgb_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.img_s, FLAGS.img_s, 3), 
            name='rgb_placeholder')
    dep_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.img_s, FLAGS.img_s, 3), 
            name='dep_placeholder')
    labels_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.n_classes),
            name='labels_placeholder')
    keep_prob_ph = tf.placeholder(tf.float32, shape=(), name='keep_prob_placeholder')
    return rgb_ph, dep_ph, labels_ph, keep_prob_ph


def fill_feed_dict(img_batch, lbl_batch, images_ph, labels_ph, keep_prob_ph, is_training):
	return


def do_eval(sess, logits, eval_correct, images_ph, labels_ph, keep_prob_ph, all_data, all_labels):
	return


#==================================================================================================
def run_training():
	# load data------------------------------------------------------------------------------------

	# tensorflow variables and operations----------------------------------------------------------
	print 'Preparing tensorflow...'
    rgb_ph, dep_ph, labels_ph, keep_prob_ph = placeholder_inputs(FLAGS.batch_size)

    logits = model.inference(rgb_ph, dep_ph, rgb_model, dep_model, keep_prob_ph)
    loss = model.loss(logits, labels_ph)
    train_op = model.training(loss)
    eval_correct = model.evaluation(logits, labels_ph)
    init_op = tf.initialize_all_variables()

	return


#==================================================================================================
def main(argv=None):
	with tf.Graph().as_default():
		run_training()
	return


if __name__ == '__main__':
	tf.app.run(main