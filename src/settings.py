import tensorflow as tf


tf.app.flags.DEFINE_string('train_dir', 'model/train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('eval_dir', 'model/eval/',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_string('experiment', '',
                           """Directory where to write event logs """
                           """and checkpoint.""")


if(tf.app.flags.FLAGS.experiment == 'retina'):

    tf.app.flags.DEFINE_string('training_file', 'retinal_dataset.csv',
                               """CSV with training data""")
    tf.app.flags.DEFINE_string('training_seq', 'training.csv',
                               """CSV with training data""")

    tf.app.flags.DEFINE_string('training_half', 'first',
                               """Split data""")
    tf.app.flags.DEFINE_string('eval_file', 'retinal_dataset.csv',
                               """CSV with training data""")

    tf.app.flags.DEFINE_string('eval_half', 'second',
                               """Split data""")
    tf.app.flags.DEFINE_integer('eval_examples', 578,
                                """Number of examples to run.""")

    tf.app.flags.DEFINE_integer('train_batch_size', 4,
                                """training batch size""")
    tf.app.flags.DEFINE_integer('eval_batch_size', 4,
                                """training batch size""")

    tf.app.flags.DEFINE_integer('max_steps', 50000,
                                """Number of batches to run.""")
    tf.app.flags.DEFINE_integer('num_classes', 4,
                                """Number of classes.""")
    tf.app.flags.DEFINE_integer('num_objects', 1,
                                """Number of classes.""")

    tf.app.flags.DEFINE_integer('image_height', 480,
                                """Number of classes.""")
    tf.app.flags.DEFINE_integer('image_width', 640,
                                """Number of classes.""")
    tf.app.flags.DEFINE_float('resize_factor', 1,
                              """Number of classes.""")
    tf.app.flags.DEFINE_integer('num_channels', 3,
                                """Number of channels in input image.""")
    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Whether to log device placement.""")
    tf.app.flags.DEFINE_boolean('use_fp16', False,
                                """Train the model using fp16.""")
    tf.app.flags.DEFINE_integer('num_examples_per_epoch_train', 578,
                                """Whether to log device placement.""")
    tf.app.flags.DEFINE_integer('num_epochs_per_decay', 15,
                                """Whether to log device placement.""")
    tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3,
                              """Whether to log device placement.""")
    tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                              """Whether to log device placement.""")
    tf.app.flags.DEFINE_float('moving_average_decay', 0.999,
                              """Whether to log device placement.""")
    tf.app.flags.DEFINE_float('var', 10.0,
                              """Whether to log device placement.""")

elif (tf.app.flags.FLAGS.experiment == "laparoscopy"):

    tf.app.flags.DEFINE_integer('max_steps', 50000,
                                """Number of batches to run.""")
    tf.app.flags.DEFINE_integer('num_classes', 5,
                                """Number of classes.""")
    tf.app.flags.DEFINE_integer('num_objects', 4,
                                """Number of classes.""")

    tf.app.flags.DEFINE_string('training_file', 'training_laparoscopy.csv',
                               """CSV with training data""")
    tf.app.flags.DEFINE_string('training_half', 'all',
                               """Split data""")
    tf.app.flags.DEFINE_string('eval_file', 'eval_laparoscopy.csv',
                               """CSV with eval data""")

    tf.app.flags.DEFINE_string('eval_half', 'all',
                               """Split data""")
    # tf.app.flags.DEFINE_integer('eval_examples', 1218,
    #                             """Number of examples to run.""")
    tf.app.flags.DEFINE_integer('eval_examples', 910,
                                """Number of examples to run.""")
    tf.app.flags.DEFINE_integer('image_height', 576,
                                """Number of classes.""")
    tf.app.flags.DEFINE_integer('image_width', 720,
                                """Number of classes.""")
    tf.app.flags.DEFINE_float('resize_factor', 640 / 720,
                              """Number of classes.""")

    tf.app.flags.DEFINE_integer('num_channels', 3,
                                """Number of channels in input image.""")
    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Whether to log device placement.""")
    tf.app.flags.DEFINE_boolean('use_fp16', False,
                                """Train the model using fp16.""")
    tf.app.flags.DEFINE_integer('num_examples_per_epoch_train', 500,
                                """Whether to log device placement.""")
    tf.app.flags.DEFINE_integer('num_epochs_per_decay', 25,
                                """Whether to log device placement.""")
    tf.app.flags.DEFINE_float('initial_learning_rate', 1e-4,
                              """Whether to log device placement.""")
    tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                              """Whether to log device placement.""")
    tf.app.flags.DEFINE_float('moving_average_decay', 0.999,
                              """Whether to log device placement.""")
    tf.app.flags.DEFINE_float('var', 10.0,
                              """Whether to log device placement.""")
