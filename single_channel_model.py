import os, ipdb
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding='VALID', group=1):
    """Wrapper for TensorFlow's 2D convolution

    Args:
        input: input data
        kernel: kernel's weight
        biases: kernel's bias
        k_h: kernel's height
        k_s: kernel's width
        c_o: number of kernels
        s_h: stride's height
        s_w: stride's width

    Returns:
        Result of convolution
    """
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1,s_h,s_w,1], padding=padding)

    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i,k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    #result = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    result = tf.nn.bias_add(conv, biases)
    return result


def inference(images, net_data, keep_prob):
    """Build the inference for one single channel.

    Args:
        images: image placeholder

    Returns:
    """
    batch_size = 0 if images.get_shape()[0].value is None else images.get_shape()[0].value

    # conv-1 layer
    ## conv(11,11,96,4,4,padding='VALID',name='conv1')
    with tf.name_scope('conv1') as scope:
        conv1W = tf.Variable(net_data['conv1'][0], name='weight')
        conv1b = tf.Variable(net_data['conv1'][1], name='biases')
        conv1_in = conv(images, conv1W, conv1b, 11, 11, 96, 4, 4, padding='SAME', group=1)
        conv1 = tf.nn.relu(conv1_in, name=scope)
    ## lrn(2,2e-05,0.75,name='norm1')
    lrn1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0, name='norm1')
    ## max_pool(3,3,2,2,padding='VALID',name='pool1')
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool1')


    # conv-2 layer
    ## conv(5,5,256,1,1,group=2,name='conv2')
    with tf.name_scope('conv2') as scope:
        conv2W = tf.Variable(net_data['conv2'][0], name='weight')
        conv2b = tf.Variable(net_data['conv2'][1], name='biases')
        conv2_in = conv(maxpool1, conv2W, conv2b, 5, 5, 256, 1, 1, padding='SAME', group=2)
        conv2 = tf.nn.relu(conv2_in, name=scope)
    ## lrn(2,2e-05,0.75,name='norm2')
    lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0, name='norm2')
    ## max_pool(3,3,2,2,padding='VALID',name='pool2')
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool2')


    # conv-3 layer
    ## conv(3,3,384,1,1,name='conv3')
    with tf.name_scope('conv2') as scope:
        conv3W = tf.Variable(net_data['conv3'][0], name='weight')
        conv3b = tf.Variable(net_data['conv3'][1], name='biases')
        conv3_in = conv(maxpool2, conv3W, conv3b, 3, 3, 384, 1, 1, padding='SAME', group=1)
        conv3 = tf.nn.relu(conv3_in, name=scope)


    # conv-4 layer
    ## conv(3,3,384,1,1,group=2,name='conv4')
    with tf.name_scope('conv4') as scope:
        conv4W = tf.Variable(net_data['conv4'][0], name='weight')
        conv4b = tf.Variable(net_data['conv4'][1], name='biases')
        conv4_in = conv(conv3, conv4W, conv4b, 3, 3, 384, 1, 1, padding='SAME', group=2)
        conv4 = tf.nn.relu(conv4_in, name=scope)


    # conv-5 layer
    ## conv(3,3,256,1,1,group=2,name='conv5')
    with tf.name_scope('conv5') as scope:
        conv5W = tf.Variable(net_data['conv5'][0], name='weight')
        conv5b = tf.Variable(net_data['conv5'][1], name='biases')
        conv5_in = conv(conv4, conv5W, conv5b, 3, 3, 256, 1, 1, padding='SAME', group=2)
        conv5 = tf.nn.relu(conv5_in, name=scope)
    ## max_pool(3,3,2,2,padding='VALID',name='pool5')
    maxpool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID', name='pool5')


    # fc6 layer
    ## fc(4096, name='fc6')
    with tf.name_scope('fc6') as scope:
        fc6W = tf.Variable(net_data['fc6'][0], name='weight')
        fc6b = tf.Variable(net_data['fc6'][1], name='biases')
        fc6_in = tf.reshape(maxpool5, [batch_size, int(np.prod(maxpool5.get_shape()[1:]))])
        fc6 = tf.nn.relu_layer(fc6_in, fc6W, fc6b, name=scope)
        fc6_drop = tf.nn.dropout(fc6, keep_prob=keep_prob, name='drop')


    # fc7 layer
    ## fc(4096, name='fc7')
    with tf.name_scope('fc7') as scope:
        fc7W = tf.Variable(net_data['fc7'][0], name='weight')
        fc7b = tf.Variable(net_data['fc7'][1], name='biases')
        fc7 = tf.nn.relu_layer(fc6_drop, fc7W, fc7b, name=scope)
        fc7_drop = tf.nn.dropout(fc7, keep_prob=keep_prob, name='drop')


    # fc8 layer - classifier
    ## fc(1000, relu=False, name='fc8')
    with tf.name_scope('fc8') as scope:
        # do not use net_data as we have differenet number of classes here
        #fc8W = tf.Variable(net_data['fc8'][0], name='weight')
        #fc8b = tf.Variable(net_data['fc8'][1], name='biases')
        fc8W_mean = np.mean(net_data['fc8'][0])
        fc8W_std  = np.std(net_data['fc8'][0])
        fc8b_mean = np.mean(net_data['fc8'][1])
        fc8b_std  = np.std(net_data['fc8'][1])
        fc8W = tf.Variable(tf.random_normal([4096,FLAGS.n_classes], 
            mean=fc8W_mean, stddev=fc8W_std), name='weight')
        fc8b = tf.Variable(tf.random_normal([FLAGS.n_classes],
            mean=fc8b_mean, stddev=fc8b_std), name='biases')
        fc8 = tf.nn.xw_plus_b(fc7_drop, fc8W, fc8b, name=scope)


    # prob
    ## softmax(name='prob')
    logits = tf.nn.softmax(fc8, name='prob')
    return logits


def loss(logits, labels):
    """Return the loss as categorical cross-entropy

    Args:
        logits: probability from inference
        labels: binary sequence, where 1 means the correct class, 0 otherwise

    Returns:
        loss: categorical crossentropy loss
    """
    if labels.get_shape()[0].value is not None:
        loss = -tf.reduce_sum(labels * tf.log(logits), reduction_indices=1, name='loss')
    else:
        loss = tf.Variable(0, dtype=tf.float32, name='loss')
    return loss


def training(loss, learning_rate=None):
    """Sets up the training ops.

    Args:
        loss: loss tensor, from loss()

    Returns:
        train_op: the op for training
    """
    if learning_rate is None:
        learning_rate = FLAGS.learning_rate

    # Add a scalar summary for the snapshot loss
    tf.scalar_summary(loss.op.name, loss)

    # Create the optimizer with given learning rate
    optimizer = tf.train.AdagradOptimizer(learning_rate)

    # Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Use the optimzer to minimize the loss
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    if labels.get_shape()[0].value is None:
        return 0.0

    num_labels = tf.argmax(labels, dimension=1) # convert from binary sequences to class id
    correct = tf.nn.in_top_k(logits, num_labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
