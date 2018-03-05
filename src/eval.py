from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import itertools

from input import eval_inputs_files
from model import create_conv_net
from loss import loss_function
from loss import evaluation
from loss import precision
from loss import ml_estimation
import settings

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('eval_interval_secs', 30,
                            """How often to run the eval.""")

tf.app.flags.DEFINE_boolean('run_once',
                            False,
                            """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Whether to log device placement.""")


def eval_once(saver, summary_writer, loss, eval, image, coords_est, coords_gt, summary_op, old_step, precision, objects_est, labels_eval, logits, objects_labels):

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(
            FLAGS.train_dir + FLAGS.experiment + "/")
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading model form Checkpoint")
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print("Step:" + global_step)
    else:
        print('No checkpoint file found')
        global_step = -1
        return global_step
    if int(global_step) > int(old_step):
        # Start the queue runners.
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Starting queue runners")
        coord = tf.train.Coordinator()
        try:
            coord = tf.train.Coordinator()
            print("Train Coordinator done...")
            print("Starting Queue Runner")
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            num_iter = int(math.ceil(FLAGS.eval_examples / FLAGS.batch_size))

            num_iter * FLAGS.batch_size
            step = 0
            loss_tot = []
            joint_accuarcy_tot = np.empty(
                (0, FLAGS.num_classes * FLAGS.num_objects), float)
            objects_correct_tot = np.empty((0, FLAGS.num_objects), float)
            while step < num_iter and not coord.should_stop():
                start_time = time.time()
                loss_value, eval_value, image_value, coords_eval_value, coords_gt_value, precision_value, objects_est_value, labels_eval_value, logits_value, objects_labels_value = sess.run(
                    [loss, eval, image, coords_est, coords_gt, precision, objects_est, labels_eval, logits, objects_labels])

                duration = time.time() - start_time
                loss_tot = np.append(loss_tot, loss_value)

                raw_image = image_value[:, :, :, 3:6]

                # plt.figure(num=None, figsize=(20, 8), dpi=200, facecolor='w', edgecolor='k')
                # for o in range(0,   FLAGS.num_classes):
                #     ax1 = plt.subplot(FLAGS.num_objects+1,FLAGS.num_classes,o+1)
                #     ax1.imshow(raw_image[0, :, :, :])
                #
                # for o in range(0, FLAGS.num_objects *  FLAGS.num_classes):
                #     ax1 = plt.subplot(FLAGS.num_objects+1,FLAGS.num_classes,o+1 + FLAGS.num_classes)
                #     x = np.arange(0, 640, 1)
                #     y = np.arange(0, 512, 1)
                #     xx, yy = np.meshgrid(x, y)
                #     zz = logits_value[0, :, :, o]
                #     ux = np.sum(zz*xx) / np.sum(zz)
                #     uy = np.sum(zz*yy) / np.sum(zz)
                #     # coords_eval_value[0,o,0] = uy
                #     # coords_eval_value[0,o,1] = ux
                #     ox = np.sqrt(np.sum(zz*(xx-ux)**2) / np.sum(zz))
                #     oy = np.sqrt(np.sum(zz*(yy-uy)**2) / np.sum(zz))
                #     variance = np.sqrt(ox**2 + oy**2)
                #     # variance = np.std(logits_value[0, :, :, o])
                #
                #
                #     ax1.set_title('var =' + '{0:02f}'.format(ox) + " " + '{0:02f}'.format(oy))
                #     ax1.imshow(logits_value[0, :, :, o])
                #     # ax1.imshow(variance)
                # plt.savefig('out_images/est1_' + '{0:05d}'.format(step * FLAGS.batch_size) + '.jpg')
                # plt.close()

                # Check instrument detection accuracy
                objects_est_value_binary = objects_est_value > 0.5
                objects_correct = np.abs(
                    objects_est_value_binary - objects_labels_value) > 0
                objects_correct = 1 - objects_correct
                objects_correct_tot = np.vstack(
                    (objects_correct_tot, objects_correct))

                joint_accuracy = np.ones(
                    [FLAGS.batch_size, FLAGS.num_objects * FLAGS.num_classes]) * -1e-10

                # instrument_names = ['Right Clasper', 'Left Clasper', "Right Scissor", 'Left Scissor']
                # joint_names = ['Left Tip', 'Right Tip', "Shaft Point", 'End Point', 'Head Point']

                instrument_names = [
                    'Instrument 1', 'Left Clasper', "Right Scissor", 'Left Scissor']
                joint_names = ['Start Shaft', 'End Shaft',
                               "Right Tip", 'Left Tip', 'Head Point']

                plt.figure()
                plt.subplot2grid((1, 5), (0, 0), colspan=3)
                for i in range(FLAGS.batch_size):
                    plt.imshow(raw_image[i, :, :, :])
                    colors = itertools.cycle(['b', 'r', 'm', 'y', 'c'])
                    colors2 = itertools.cycle(['b', 'r', 'm', 'y', 'c'])
                    for o in range(0, FLAGS.num_objects):

                        for p in range(0, FLAGS.num_classes):
                            idx = o * FLAGS.num_classes + p
                            color = 'r'

                            if(objects_labels_value[i, o] > 0.5):
                                plt.scatter(coords_gt_value[i, idx, 1],
                                            coords_gt_value[i, idx, 0],    color='g', s=5)

                                joint_accuracy[i, idx] = np.sqrt((coords_eval_value[i, idx, 1] - coords_gt_value[i, idx, 1])**2 + (
                                    coords_eval_value[i, idx, 0] - coords_gt_value[i, idx, 0])**2)
                                plt.text(730 + o * 100, 40 + p * 40,
                                         '{0:.2f}'.format(joint_accuracy[i, idx]), fontsize=4)

                                color = 'g'

                            if (objects_est_value[i, o] > 0.5):
                                plt.scatter(
                                    coords_eval_value[i, idx, 1], coords_eval_value[i, idx, 0], color=next(colors), s=5)

                            plt.text(650, 40 + p * 40,
                                     joint_names[p], fontsize=4, color=next(colors2))
                        plt.text(730 + o * 100, 10,
                                 instrument_names[o], fontsize=4, color=color)

                    plt.text(650, 60 + p * 50 + 75,
                             "Euclidean distance error in pixels", fontsize=4)
                    plt.text(650, 60 + p * 50 + 75 + 40,
                             "Green points denote the ground truth", fontsize=4, color='g')
                    plt.axis('off')
                    plt.savefig('out_images/image' + '{0:05d}'.format(step * FLAGS.batch_size + i + 10) +
                                '.png', transparent=True, bbox_inches='tight', pad_inches=0.1, dpi=385)
                    plt.close()

                # print(joint_accuracy)
                joint_accuarcy_tot = np.vstack(
                    (joint_accuarcy_tot, joint_accuracy))

                format_str = (
                    '%s: step %d, loss = %.6f, precision=%.6f, accuracy = %.3f (%.1f examples/sec; %.3f ''sec/batch)')
                eval_examples_per_step = FLAGS.batch_size
                examples_per_sec = eval_examples_per_step / duration
                sec_per_batch = float(duration)

                print_str_loss = format_str % (datetime.now(), step, loss_value, precision_value, np.mean(
                    eval_value), examples_per_sec, sec_per_batch)
                # print (print_str_loss)
                # print(eval_value)
                step += 1

            loss_tot = np.mean(loss_tot)
            # Percentage Detected Plot
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            a = 0
            perc_tot = np.empty(
                (0, FLAGS.num_classes * FLAGS.num_objects), float)
            for x in range(5, 41, 1):
                perc = (np.sum((joint_accuarcy_tot <= float(x)), axis=0) - np.sum((joint_accuarcy_tot <
                                                                                   0.0), axis=0)) / np.sum((joint_accuarcy_tot >= 0.0), axis=0) * 100
                perc_tot = np.vstack((perc_tot, perc))
                a = a + 1
                print(x)
                print(perc)

            for o in range(0, FLAGS.num_objects):
                for p in range(0, FLAGS.num_classes):
                    print("Joint_" + str(o + 1) + "_" +
                          str(p + 1) + "= np.array([")
                    a = 0
                    for x in range(5, 31, 5):
                        print(
                            "[" + str(x) + ", " + str(perc_tot[a, o * FLAGS.num_classes + p]) + "],")
                        a = a + 1

                    print("])")

            joint_accuarcy_tot = (np.sum((joint_accuarcy_tot)) -
                                  np.sum((joint_accuarcy_tot < 0.0))) / np.sum((joint_accuarcy_tot >= 0.0))
            preciscion = np.mean(objects_correct_tot)

            print(np.mean(objects_correct_tot, axis=0))
            print('%s: loss @ %s = %.3f, joint accuarcy = %.3f, instrument detection = %.3f' %
                  (datetime.now(), global_step, loss_tot, np.mean(joint_accuarcy_tot), preciscion))

            # for e in range(0, FLAGS.num_classes*FLAGS.num_objects):
            #     summary.value.add(tag='accuracy_'+str(e), simple_value=joint_accuarcy_tot[e])

            summary.value.add(tag='preciscion', simple_value=preciscion)
            summary_writer.add_summary(summary, global_step)

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
    else:
        print("Step already computed")
    return global_step


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:

        var = FLAGS.var
        channels = FLAGS.num_channels
        FLAGS.num_classes

        images_eval, labels_eval, labels_objects = eval_inputs_files(
            filename=FLAGS.eval_file, data_dir="../data/retina/", batch_size=FLAGS.batch_size, var=var, half=FLAGS.eval_half)
        bn_training_placeholder = False

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, logits_objects = create_conv_net(images_eval[:, :, :, 0:3], 1.0,  channels, FLAGS.num_classes *
                                                 FLAGS.num_objects, summaries=False, two_sublayers=False, training=bn_training_placeholder, dropout=1.0)
        # logits_objects = alexnet(images_eval[:,:,:,0:3], 1.0)
        # logits = logits_objects
        # Calculate predictions.
        loss_ = loss_function(logits, logits_objects,
                              labels_eval, labels_objects)
        evaluation_ = evaluation(logits, labels_eval, False)
        precision_ = precision(logits_objects, labels_objects)
        coords_est = ml_estimation(logits)
        coords_gt = ml_estimation(labels_eval)

        logits = tf.transpose(logits, [0, 3, 1, 2])  # b,h,w,j to b,j,h,w
        logits = tf.reshape(logits, [FLAGS.batch_size * FLAGS.num_classes * FLAGS.num_objects, tf.cast(
            FLAGS.image_height * FLAGS.image_width * FLAGS.resize_factor * FLAGS.resize_factor, tf.int32)])
        logits = tf.nn.softmax(logits)
        # logits = logits - (tf.expand_dims(tf.reduce_min(logits,axis=1),axis=1)) # min = 0
        # logits = logits / (tf.expand_dims(tf.reduce_sum(logits,axis=1),axis=1)) # sum = 1
        logits = tf.reshape(logits, [FLAGS.batch_size, FLAGS.num_classes * FLAGS.num_objects, tf.cast(
            FLAGS.image_height * FLAGS.resize_factor, tf.int32), tf.cast(FLAGS.image_width * FLAGS.resize_factor, tf.int32)])
        logits = tf.transpose(logits, [0, 2, 3, 1])

        # Restore the moving average version of the learned variables for eval.
        # variable_averages = tf.train.ExponentialMovingAverage(0.999)
        # variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver()

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(
            FLAGS.eval_dir + FLAGS.experiment, g)
        old_step = -1
        while int(old_step) < (FLAGS.max_steps - 1):
            old_step = eval_once(saver, summary_writer, loss_, evaluation_, images_eval, coords_est, coords_gt,
                                 summary_op, old_step, precision_, tf.nn.sigmoid(logits_objects), labels_eval, logits, labels_objects)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
