from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf
from collections import OrderedDict
from layers import *

FLAGS = tf.app.flags.FLAGS


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % "TOWER_NAME", '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_conv_net(x, keep_prob, channels, n_class, layers=6, features_root=64, filter_size=3, pool_size=2, summaries=True, two_sublayers=True, training=True, classification=False, dropout=1.0):
    """
    Creates a new convolutional unet for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    :param two_sublayers: Use two sublayers per layer
    :param training: Training or testing phase
    :param classification: Enable classification output

    """

    print("Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(layers=layers,
                                                                                                                                     features=features_root,
                                                                                                                                     filter_size=filter_size,
                                                                                                                                     pool_size=pool_size))
    # Placeholder for the input image
    tf.shape(x)[1]
    tf.shape(x)[2]
    #x_image = tf.reshape(x, tf.pack([-1,nx,ny,channels]))
    in_node = x
    #batch_size = tf.shape(x_image)[0]

    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()

    in_size = 1000
    size = in_size
    # down layers
    for layer in range(0, layers):
        features = 2**layer * features_root
        stddev = tf.sqrt(2 / (filter_size**2 * features))
        if layer == 0:
            w1 = weight_variable(
                [filter_size, filter_size, channels, features], stddev)
        else:
            w1 = weight_variable(
                [filter_size, filter_size, features // 2, features], stddev)

        b1 = bias_variable([features])
        if two_sublayers == True:
            w2 = weight_variable(
                [filter_size, filter_size, features, features], stddev)
            b2 = bias_variable([features])

        if two_sublayers == True:
            conv1 = conv2d(in_node, w1, keep_prob)
            tmp_h_conv = tf.nn.relu(conv1 + b1)
            conv2 = conv2d(tmp_h_conv, w2, keep_prob)
            dw_h_convs[layer] = tf.nn.relu(conv2 + b2)

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

        else:
            conv1 = conv2d(in_node, w1, keep_prob)
            print(in_node)
            print(conv1)
            dw_h_convs[layer] = tf.contrib.layers.batch_norm(
                inputs=conv1 + b1, decay=0.9, is_training=training, center=True, scale=True, activation_fn=tf.nn.relu, updates_collections=None)
            # dw_h_convs[layer] = tf.nn.relu(conv1 + b1)
            # dw_h_convs[layer] = tf.nn.dropout(activate, dropout)

            #dw_h_convs[layer] = conv1
            weights.append((w1))
            biases.append((b1))
            convs.append((conv1))

        size -= 4
        if layer < layers - 1:
            pools[layer] = max_pool(dw_h_convs[layer], pool_size)
            in_node = pools[layer]
            size /= 2

    in_node = dw_h_convs[layers - 1]

    with tf.variable_scope('local3') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
        w1 = weight_variable(
            [filter_size, filter_size, features, features // 8], stddev)
        b1 = bias_variable([features // 8])
        conv_bottle = conv2d(in_node, w1, keep_prob)
        relu_bottle = tf.nn.relu(conv_bottle + b1)
        # relu_bottle = max_pool(relu_bottle, pool_size)

        # w1 = weight_variable([filter_size, filter_size, features//8, features//16], stddev)
        # b1 = bias_variable([features//16])
        # conv_bottle = conv2d(relu_bottle, w1, keep_prob)
        # relu_bottle = tf.nn.relu(conv_bottle + b1)
        # relu_bottle = max_pool(relu_bottle, pool_size)

        reshape = tf.reshape(relu_bottle, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = weight_variable(shape=[dim, 384], stddev=0.04)
        biases = bias_variable([384])
        local3 = tf.nn.relu(tf.matmul(reshape, weights) +
                            biases, name=scope.name)
        local3 = tf.nn.dropout(local3, dropout)

  # local4
    with tf.variable_scope('local4') as scope:
        weights = weight_variable(shape=[384, 192], stddev=0.04)
        biases = bias_variable([192])
        local4 = tf.nn.relu(tf.matmul(local3, weights) +
                            biases, name=scope.name)
        local4 = tf.nn.dropout(local4, dropout)

        weights = weight_variable([192, FLAGS.num_objects], stddev=1 / 192)
        biases = bias_variable([FLAGS.num_objects])
        out_class = tf.matmul(local4, weights) + biases

        # up layers
    for layer in range(layers - 2, -1, -1):
        features = 2**(layer + 1) * features_root
        stddev = tf.sqrt(2 / (filter_size**2 * features))

        wd = weight_variable_deconv(
            [pool_size, pool_size, features // 2, features], stddev)
        bd = bias_variable([features // 2])
        h_deconv = deconv2d(in_node, wd, pool_size) + bd
        print(h_deconv)
        h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)

        deconv[layer] = h_deconv_concat

        w1 = weight_variable(
            [filter_size, filter_size, features, features // 2], stddev)
        b1 = bias_variable([features // 2])

        if two_sublayers == True:
            w2 = weight_variable(
                [filter_size, filter_size, features // 2, features // 2], stddev)
            b2 = bias_variable([features // 2])

        if two_sublayers == True:
            conv1 = conv2d(h_deconv_concat, w1, keep_prob)
            h_conv = tf.nn.relu(conv1 + b1)
            conv2 = conv2d(h_conv, w2, keep_prob)
            in_node = tf.nn.relu(conv2 + b2)
            up_h_convs[layer] = in_node

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))
        else:
            conv1 = conv2d(h_deconv_concat, w1, keep_prob)
            print(conv1)
            in_node = tf.contrib.layers.batch_norm(
                inputs=conv1 + b1, decay=0.9, is_training=training, center=True, scale=True, activation_fn=tf.nn.relu, updates_collections=None)
            # in_node = tf.nn.relu(conv1 + b1)
            # in_node = tf.nn.dropout(in_node, dropout)
            #in_node = conv1
            up_h_convs[layer] = in_node

            # weights.append((w1))
            # biases.append((b1))
            # convs.append((conv1))

        size *= 2
        size -= 4

    # Output Map
    stddev = tf.sqrt(2 / (1**2 * features_root))
    weight = weight_variable([1, 1, features_root, n_class], stddev)
    bias = bias_variable([n_class])
    conv = conv2d(in_node, weight, tf.constant(1.0), pad=False)
    output_map = conv + bias
    up_h_convs["out"] = output_map
    # out_class = tf.zeros([FLAGS.batch_size,2])
    return output_map, out_class


def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """

    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.pack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.pack((-1, img_w, img_h, 1)))
    return V


def train_model(total_loss, global_step):
    """Train model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = FLAGS.num_examples_per_epoch_train / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    return train_op


def _add_loss_summaries(total_loss):
    """Add summaries for losses

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    # with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(
        name, shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
