from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import h5py
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import math

FLAGS = tf.app.flags.FLAGS
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000


def _generate_image_and_label_batch(image, label, num_instrument,  min_queue_examples,
                                    batch_size, shuffle, train=True, num_preprocess_threads=5):
    if shuffle:
        images, label_batch, num_instrument_batch = tf.train.shuffle_batch(
            [image, label, num_instrument],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 2 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch, num_instrument_batch = tf.train.batch(
            [image, label, num_instrument],
            batch_size=batch_size,
            num_threads=1,
            capacity=min_queue_examples + 2 * batch_size)

  # Display the training images in the visualizer.
  #   if train == True:
  #     tf.image_summary('images_train', images)
  # # else:
  # #     tf.image_summary('images_eval', images)
    return images, label_batch, num_instrument_batch


def balance_classes(class_list):
    classes = np.unique(class_list)
    num_entries = np.zeros([classes.size, 1])
    for idx in range(0, classes.size):
        num_entries[idx] = np.sum(class_list == classes[idx])
        print(num_entries)
    max_entries = np.max(num_entries)
    print("Class balancer max entries:")
    print(max_entries)
    for idx in range(0, classes.size):
        idx_before = np.argwhere(class_list == classes[idx])

        new_entry_count = np.int32(max_entries - num_entries[idx])

        idx_new = np.random.choice(np.squeeze(idx_before), size=[
                                   new_entry_count[0], 1], replace=True)
        idx_new = np.vstack((idx_before, idx_new))
        print("Class balancer new entries:")
        print(idx_new.shape)
        if idx == 0:
            idx_tot = idx_new
        else:
            idx_tot = np.vstack((idx_tot, idx_new))

    return np.squeeze(idx_tot.astype(np.int32))


def load_csv(filename, half='all', sequences=[0, 1, 2, 3, 4, 5]):
    data = np.genfromtxt(filename, delimiter=',', dtype=str)
    sequences = np.unique(data[:, 1])
    print(data.shape)
    for seq in sequences:
        if np.any(sequences == seq):
            print(seq)
            # get indexes of sequence
            idx = np.array(np.where(data[:, 1] == seq))
            print(idx.shape)
            if half == 'first':
                half_val = int(np.floor(idx.shape[1] / 2))
                idx = idx[0, :half_val]
            elif half == 'second':
                half_val = int(np.floor(idx.shape[1] / 2))
                idx = idx[0, half_val:]
            idx = np.squeeze(idx)
            idx = np.expand_dims(idx, axis=1)

            try:

                idx_tot = np.vstack((idx_tot, idx))
            except NameError:
                idx_tot = idx

    print(idx_tot.shape)
    idx_tot = np.squeeze(idx_tot)
    image_filenames = data[idx_tot, 0]
    num_instruments = data[idx_tot, 2:2 +
                           FLAGS.num_objects].astype(np.float32)
    pos = data[idx_tot, 2 + FLAGS.num_objects:].astype(np.float32)
    pos_shape = pos.shape
    print(pos_shape)
    pos = np.reshape(pos, (pos.shape[0], int(pos.shape[1] / 2), 2))
    print(pos.shape)
    # if half == 'first':
    #     print("Using first half of the data set")
    #     half = int(np.floor(image_filenames.shape[0]/2))
    #     image_filenames = image_filenames[:half]
    #     num_instruments = num_instruments[:half]
    #     pos = pos[:half, :, :]
    # elif half == 'second':
    #     print("Using second half of the data set")
    #     half = int(np.floor(image_filenames.shape[0]/2))
    #     image_filenames = image_filenames[half:]
    #     num_instruments = num_instruments[half:]
    #     pos = pos[half:, :, :]
    print("Image filenames shape:")
    print(image_filenames.shape)
    return image_filenames, pos, num_instruments
    #


def distorted_inputs_files(filename, data_dir, batch_size, var, half='both'):
    """Construct distorted input for CIFAR training using the Reader ops.

    Args:
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """

    with tf.name_scope('input'):
        print("Loading image filenames")
        filenames, coords, num_instruments = load_csv(
            data_dir + filename, half)
        power_two = [1 << i for i in range(FLAGS.num_objects)]
        idx = balance_classes(np.sum(num_instruments * power_two, axis=1))
        filenames = filenames[idx]
        coords = coords[idx, :]
        num_instruments = num_instruments[idx, :]

    #   filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
    #   coords = tf.convert_to_tensor(coords, dtype=tf.float32)

    with tf.device('/cpu:0'):
        queue = tf.train.slice_input_producer(
            [filenames, coords, num_instruments],  name='string_DISTORTED_input_producer')

        image_filename = queue[0]
        coord = queue[1]
        num_instrument = queue[2]

        # image_filename = tf.Print(image_filename,[image_filename, num_instrument],"")

        file_content = tf.read_file(data_dir + image_filename)

        image = tf.image.decode_png(file_content)
        image = tf.image.resize_images(image, [tf.cast(FLAGS.image_height * FLAGS.resize_factor, tf.int32), tf.cast(
            FLAGS.image_width * FLAGS.resize_factor, tf.int32)], method=0, align_corners=False)

        image.set_shape([FLAGS.image_height * FLAGS.resize_factor,
                         FLAGS.image_width * FLAGS.resize_factor, FLAGS.num_channels])
        image = tf.cast(image, tf.float32)  # / 255.0

        num_instrument = tf.cast(num_instrument, tf.float32)
        coord = coord * FLAGS.resize_factor

        # get list before rotation
        coord_navail = tf.cast(coord[:, 0] < 0, tf.float32)

        # # # # # Up Down Flip
        # flip = tf.random_uniform([1, 1], minval=0, maxval=1, dtype=tf.float32)
        # image = tf.cond(flip[0][0] > 0.5, lambda: tf.image.flip_up_down(image), lambda: image)
        #
        # delta_coords = tf.cond(flip[0][0] > 0.5, lambda:  tf.constant([[FLAGS.image_height * FLAGS.resize_factor,0.0]]), lambda: tf.constant([[0.0,0.0]]))
        #
        # coord = tf.sub(tf.tile(delta_coords, [12, 1]) , coord)
        # coord = tf.abs(coord)
        # # # #
        # # flip_left_right
        # flip = tf.random_uniform([1, 1], minval=0, maxval=1, dtype=tf.float32)
        # image = tf.cond(flip[0][0] > 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
        # delta_coords = tf.cond(flip[0][0] > 0.5, lambda:  tf.constant([[FLAGS.image_width,0.0]]),
        # lambda: tf.constant([[0.0,0.0]]))
        #
        # coord = tf.sub(tf.tile(delta_coords, [12, 1]) , coord)
        # coord = tf.abs(coord)

        # #

        uniform = tf.constant(1 / (FLAGS.image_height * FLAGS.image_width * FLAGS.resize_factor * FLAGS.resize_factor), shape=[
                              FLAGS.image_height * FLAGS.resize_factor, FLAGS.image_width * FLAGS.resize_factor, FLAGS.num_classes * FLAGS.num_objects])
        print(uniform)

        label = tf.squeeze(tf.exp(log_prob_q(coord, var)), axis=0)

        # label = tf.squeeze(circle_map(coord, var, out_shape))
        label = tf.transpose(label, [1, 2, 0])
        label = label + coord_navail * uniform
        # label = tf.Print(label, [tf.reduce_sum(label)], "Sum: ")
        #
        # distorted_image = tf.image.random_brightness(image,max_delta=0.15)
        # distorted_image = tf.image.random_contrast(distorted_image,lower=0.1, upper=0.75)
        distorted_image = image
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)

        # coord = tf.unstack(coord)
        float_image = tf.image.per_image_standardization(distorted_image)
        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.8
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)
        print('Filling queue with %d images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)

        # Generate a batch of images and labels by building up a queue of examples.
        return _generate_image_and_label_batch(float_image, label, num_instrument,
                                               min_queue_examples, batch_size,
                                               shuffle=True, num_preprocess_threads=8)


def eval_inputs_files(filename, data_dir, batch_size, var, half='all'):
    with tf.name_scope('input'):
        print("Loading image filenames")
        filenames, coords, num_instruments = load_csv(
            data_dir + filename, half)

    with tf.device('/cpu:0'):

        queue = tf.train.slice_input_producer(
            [filenames, coords, num_instruments],  name='string_DISTORTED_input_producer', shuffle=False)
        image_filename = queue[0]
        coord = queue[1]
        num_instrument = queue[2]
        file_content = tf.read_file(data_dir + image_filename)

        image = tf.image.decode_png(file_content)
        image = tf.image.resize_images(image, [tf.cast(FLAGS.image_height * FLAGS.resize_factor, tf.int32), tf.cast(
            FLAGS.image_width * FLAGS.resize_factor, tf.int32)], method=0, align_corners=False)

        image.set_shape([FLAGS.image_height * FLAGS.resize_factor,
                         FLAGS.image_width * FLAGS.resize_factor, FLAGS.num_channels])
        image = tf.cast(image, tf.float32)  # / 255.0

        num_instrument = tf.cast(num_instrument, tf.float32)
        coord = coord * FLAGS.resize_factor

        uniform = tf.constant(1 / (FLAGS.image_height * FLAGS.image_width * FLAGS.resize_factor * FLAGS.resize_factor), shape=[
                              FLAGS.image_height * FLAGS.resize_factor, FLAGS.image_width * FLAGS.resize_factor, FLAGS.num_classes * FLAGS.num_objects])

        coord_navail = tf.cast(coord[:, 0] < 0, tf.float32)

        label = tf.squeeze(tf.exp(log_prob_q(coord, var)), axis=0)
        label = tf.transpose(label, [1, 2, 0])
        label = label + coord_navail * uniform

        float_image = tf.image.per_image_standardization(image)
        float_image = tf.concat(axis=2, values=[float_image, image / 255.0])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.2
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)
        print('Filling queue with %d images before starting to train. ''This will take a few minutes.' %
              min_queue_examples)
        # Generate a batch of images and labels by building up a queue of examples.
        return _generate_image_and_label_batch(float_image, label, num_instrument, min_queue_examples, batch_size, train=False, shuffle=False, num_preprocess_threads=3)


def log_prob_q(coords, var):
    grid_col, grid_row = tf.meshgrid(tf.linspace(0.0, tf.cast(FLAGS.image_width * FLAGS.resize_factor - 1, tf.float32), tf.cast(FLAGS.image_width * FLAGS.resize_factor, tf.int32)),
                                     tf.linspace(0.0, tf.cast(FLAGS.image_height * FLAGS.resize_factor - 1, tf.float32), tf.cast(FLAGS.image_height * FLAGS.resize_factor, tf.int32)))
    print(grid_col.get_shape())
    # #grid.size = (460,640)
    #
    coord_x = coords[:, 1]
    coord_y = coords[:, 0]
    # #coord_x = (minibatch_size, num_joints)
    # # replace with tf.expand_dims()
    print(tf.shape(coord_x))
    mean_x = tf.expand_dims(coord_x, 1)
    mean_x = tf.expand_dims(mean_x, 2)
    print(tf.shape(mean_x))
    mean_y = tf.expand_dims(coord_y, 1)
    mean_y = tf.expand_dims(mean_y, 2)
    #
    diff_x = grid_col[None, None, ...] - mean_x
    diff_y = grid_row[None, None, ...] - mean_y
    dist = (tf.pow(diff_x, 2) + tf.pow(diff_y, 2)) / var
    return -tf.log(2 * math.pi) - tf.log(var) - dist / 2
