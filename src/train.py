
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf
import numpy as np
import math

from input import distorted_inputs_files
from model import create_conv_net
from loss import loss_function
from model import train_model
from loss import evaluation
from loss import precision
import settings


import matplotlib as mpl
mpl.use('Agg')


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 2,
                            """Batch size""")


def train():

    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        tb_step = 50
        save_step = math.ceil(
            FLAGS.num_examples_per_epoch_train / FLAGS.batch_size)

        var = FLAGS.var

        images_train, labels_train, labels_objects = distorted_inputs_files(
            filename=FLAGS.training_file, data_dir="../data/retina/", batch_size=FLAGS.batch_size, var=var, half=FLAGS.training_half)
        print("Image_train shape: " + str(images_train.get_shape()))

        images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, int(
            FLAGS.image_height * FLAGS.resize_factor), int(FLAGS.image_width * FLAGS.resize_factor), FLAGS.num_channels))

        labels_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, int(
            FLAGS.image_height * FLAGS.resize_factor), int(FLAGS.image_width * FLAGS.resize_factor), FLAGS.num_classes * FLAGS.num_objects))

        labels_objects_placeholder = tf.placeholder(
            tf.float32, shape=(FLAGS.batch_size, FLAGS.num_objects))

        bn_training_placeholder = tf.placeholder(tf.bool)
        keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits_map, logits_class = create_conv_net(images_placeholder, keep_prob,  FLAGS.num_channels, FLAGS.num_classes *
                                                   FLAGS.num_objects, summaries=False, two_sublayers=False, training=bn_training_placeholder, dropout=0.5)

        loss_ = loss_function(logits_map, logits_class, labels_placeholder,
                              labels_objects_placeholder)
        evaluation_ = evaluation(logits_map, labels_placeholder, True)
        precision_ = precision(logits_class, labels_objects_placeholder)

        train_op = train_model(loss_, global_step)

        saver = tf.train.Saver(max_to_keep=0)  # keep all checkpoints

        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            print("Starting Training Loop")
            sess.run(tf.local_variables_initializer())
            print("Local Variable Initializer done...")
            sess.run(tf.global_variables_initializer())
            print("Global Variable Initializer done...")
            coord = tf.train.Coordinator()
            print("Train Coordinator done...")
            print("Starting Queue Runner")
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for step in range(FLAGS.max_steps):
                start_time = time.time()
                images_s, labels_s, labels_objects_s = sess.run(
                    [images_train, labels_train, labels_objects])

                train_feed = {images_placeholder: images_s,
                              labels_placeholder: labels_s,
                              labels_objects_placeholder: labels_objects_s,
                              keep_prob: 0.5,
                              bn_training_placeholder: True}

                if step % tb_step == 0:
                    summary, _, precision_value, evaluation_value, loss_value = sess.run(
                        [merged, train_op, precision_, evaluation_, loss_], feed_dict=train_feed)

                else:
                    _, precision_value, evaluation_value, loss_value = sess.run(
                        [train_op, precision_, evaluation_, loss_], feed_dict=train_feed)

                duration = time.time() - start_time

                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.6f, precision=%.6f, joint_accuarcy=%.6f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print_str_loss = format_str % (datetime.now(), step, loss_value, precision_value, np.mean(evaluation_value),
                                               examples_per_sec, sec_per_batch)
                print(print_str_loss)

                if step % save_step == 0:
                    save_path = saver.save(sess, FLAGS.train_dir + FLAGS.experiment +
                                           "/" + 'model.cpkt-' + str(step), write_meta_graph=False)
                    print('Model saved in file: ' + save_path)

            save_path = saver.save(sess, FLAGS.train_dir + FLAGS.experiment +
                                   "/" + 'model.cpkt-' + str(step), write_meta_graph=False)

            print('Model saved in file: ' + save_path)
            coord.request_stop()
            coord.join(threads)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir + FLAGS.experiment + "/"):
        tf.gfile.DeleteRecursively(FLAGS.train_dir + FLAGS.experiment + "/")
    tf.gfile.MakeDirs(FLAGS.train_dir + FLAGS.experiment + "/")
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
