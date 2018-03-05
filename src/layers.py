import tensorflow as tf


def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def weight_variable_deconv(shape, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, keep_prob_, pad=True):

    # if pad:
    #     x = tf.pad(x, [[0,0],[1, 1,], [1, 1],[0,0]], "SYMMETRIC")
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # return tf.nn.dropout(conv_2d, keep_prob_)


def deconv2d(x, W, stride, pad=False, factor=2):
    x_shape = tf.shape(x)
    factor = tf.constant(factor, tf.float32)
    channels = tf.cast(
        tf.round(tf.cast(x_shape[3], tf.float32) // factor), tf.int32)
    output_shape = tf.stack(
        [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels])
    out = tf.nn.conv2d_transpose(x, W, output_shape, strides=[
                                 1, stride, stride, 1], padding='SAME')
    print(out)
    # if pad:
    #     out = tf.pad(x, [[0,0],[1, 1,], [1, 1],[0,0]], "SYMMETRIC")
    return out


def max_pool(x, n, pad=False):
    if pad:
        x = tf.pad(x, [[0, 0], [1, 1, ], [1, 1], [0, 0]], "SYMMETRIC")
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')


def crop_and_concat(x1, x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2,
               (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)


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
