import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def evaluation(logits, labels, summary):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    print('Evaluation..')
    logits_est = ml_estimation(logits)
    labels_est = ml_estimation(labels)

    # logits_est = tf.Print(logits_est,[logits_est],"")
    # labels_est = tf.Print(labels_est,[labels_est],"")
    distance = mean_distance(logits_est, labels_est)
    # if summary:
    #     for nclass in range(FLAGS.num_classes):
    #          tf.scalar_summary('accuracy_'+str(nclass), distance[nclass])
    #
    # #     tf.scalar_summary('evaluation_1', distance[1])
    # #     tf.scalar_summary('evaluation_2', distance[2])
    # #     tf.scalar_summary('evaluation_3', distance[3])
    return distance


def precision(logits, labels):
    labels = tf.cast(labels, tf.int32)
    # labels = tf.Print(labels,[labels],"")
    logits = tf.cast(tf.nn.sigmoid(logits) > 0.5, tf.int32)
    # logits = tf.Print(logits,[logits],"logits")
    # logits = tf.cast(tf.reduce_sum(labels, axis=1),tf.int32)
    # top_k_op  = tf.nn.in_top_k(logits, labels, 1)
    correct = tf.cast(tf.abs(tf.subtract(labels, logits)) > 0, tf.float32)
    # correct = tf.Print(correct,[correct],"correct")
    return 1 - tf.reduce_mean(correct)
    # return tf.constant(1.0)


def em_estimation(prob_map):
    prob_map = tf.transpose(prob_map, [0, 3, 1, 2])  # b,h,w,j to b,j,h,w
    grid = np.mgrid[:prob_map.get_shape[2], :prob_map.get_shape[3]]
    # grid = tf.expand_dims(grid,1)
    # grid = tf.expand_dims(grid,2)
    prob_map = tf.expand_dims(prob_map, 2)
    res = tf.reduce_sum(grid[None, None] * prob_map, axis=(3, 4))
    return res


def ml_estimation(prob_map):
    prob_map = tf.transpose(prob_map, [0, 3, 1, 2])  # b,h,w,j to b,j,h,w
    aux = tf.reshape(prob_map, (FLAGS.batch_size,
                                FLAGS.num_classes * FLAGS.num_objects, -1))
    argmax = tf.argmax(aux, axis=2)
    r = argmax // tf.cast(FLAGS.image_width * FLAGS.resize_factor, tf.int64)
    c = argmax % tf.cast(FLAGS.image_width * FLAGS.resize_factor, tf.int64)
    return tf.stack([r, c], axis=2)


def mean_distance(v1, v2):
    """v1 and v2 have shape (batch_size, num_joints, 2)"""
    diffsqr = (v1 - v2) ** 2
    distsqr = tf.cast(tf.reduce_sum(diffsqr, axis=2), tf.float32)
    dist = tf.sqrt(distsqr)
    return tf.reduce_mean(dist, axis=0)


def loss_function(logits_maps, logits_tools, labels_maps, labels_tools):
    """Combined loss function for heatmaps and tool classification

    Args:
      logits_maps: Heat map logits from inference
      logits_tools: Tool logits from inference
      labels_maps: Heat map labels
      labels_tools: Tool labels
    Returns:
      Loss tensor of type float.
    """
    with tf.variable_scope('total_loss') as scope:
        logits_maps = tf.transpose(
            logits_maps, [0, 3, 1, 2])  # b,h,w,j to b,j,h,w
        logits_maps = tf.reshape(logits_maps, [FLAGS.batch_size * FLAGS.num_classes * FLAGS.num_objects, tf.cast(
            FLAGS.image_height * FLAGS.image_width * FLAGS.resize_factor * FLAGS.resize_factor, tf.int32)])
        logits_maps = tf.nn.softmax(logits_maps)

        logits_maps = tf.reshape(logits_maps, [FLAGS.batch_size, FLAGS.num_classes * FLAGS.num_objects, tf.cast(
            FLAGS.image_height * FLAGS.resize_factor, tf.int32), tf.cast(FLAGS.image_width * FLAGS.resize_factor, tf.int32)])

        logits_maps = tf.transpose(logits_maps, [0, 2, 3, 1])
        logits_maps = tf.maximum(1e-15, logits_maps)

        # / (FLAGS.num_classes * FLAGS.num_objects)
        cross_entropy_mean_map = - \
            tf.reduce_sum(tf.multiply(
                labels_maps, tf.log(logits_maps)), axis=[1, 2, 3])
        tf.add_to_collection(
            'losses_map', tf.reduce_mean(cross_entropy_mean_map))

        labels = labels_tools
    #   labels = tf.Print(labels,[labels],"GT:")
    #   logits_tools = tf.Print(logits_tools,[logits_tools],"EST:")
        cross_entropy_objects = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits_tools, labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean_objects = tf.reduce_sum(
            cross_entropy_objects, name='cross_entropy', axis=[1])

        tf.add_to_collection('losses_objects', tf.reduce_mean(
            cross_entropy_mean_objects))

        cross_entropy_mean_tot = tf.reduce_sum(
            cross_entropy_mean_map + cross_entropy_mean_objects) / FLAGS.batch_size
        tf.add_to_collection('losses', cross_entropy_mean_tot)

        return tf.add_n(tf.get_collection('losses'), name='total_loss')
