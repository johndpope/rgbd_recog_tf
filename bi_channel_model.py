import os
import tensorflow as tf
import numpy as np
import single_channel_model as base_model

FLAGS = tf.app.flags.FLAGS

# alias functions
conv = base_model.conv
loss = base_model.loss
training = base_model.training


def inference(rgb_img, dep_img, keep_prob):
	return
