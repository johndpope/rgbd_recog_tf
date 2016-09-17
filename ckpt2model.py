import tensorflow as tf
import configure as cfg
import os, ipdb



def main():
	dummy = tf.Variable(0)  # dummy variable !!!
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