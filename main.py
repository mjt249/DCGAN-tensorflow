import os
import scipy.misc
import numpy as np
import sys
import socket
import os.path


from model import SDFGAN
from utils import pp, show_all_variables, create_samples

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("d_learning_rate", 0.0002, "Learning rate of discrim for adam [0.00005]")
flags.DEFINE_float("g_learning_rate", 0.0005, "Learning rate of gen for adam [0.0025]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_depth", 64, "The size of sdf field to use (will be center cropped). [64]")
flags.DEFINE_integer("input_height", None,
                     "The size of sdf to use (will be center cropped). If None, same value as input_depth [None]")
flags.DEFINE_integer("input_width", None,
                     "The size of sdf to use (will be center cropped). If None, same value as input_depth [None]")
flags.DEFINE_integer("output_depth", 64, "The size of the output sdf to produce [64]")
flags.DEFINE_integer("output_height", None,
                     "The size of the output images to produce. If None, same value as output_depth [None]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_depth [None]")
flags.DEFINE_integer("c_dim", 1, "Dimension of sdf. [1]")
flags.DEFINE_string("dataset", "shapenet", "The name of dataset [shapenet]")
flags.DEFINE_string("input_fname_pattern", "*.npy", "Glob pattern of filename of input sdf [*]")
flags.DEFINE_string("checkpoint_dir", None, "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("dataset_dir", "data", "Directory name to read the input training data [data]")
flags.DEFINE_string("log_dir", "logs", "Directory name to save the log files [logs]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_string("job_name", None, "ps or worker")
flags.DEFINE_integer("task_index", None, "the id of the task")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_height is None:
        FLAGS.input_height = FLAGS.input_depth
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_depth

    if FLAGS.output_height is None:
        FLAGS.output_height = FLAGS.output_depth
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_depth

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)



    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    if FLAGS.job_name == None:
        raise ValueError("invalid job_name argument")

    if FLAGS.task_index == None:
        raise ValueError("invalid task_index argument")

    if FLAGS.checkpoint_dir == None:
        raise ValueError("invalid checkpoint_dir argument")

    cluster = tf.train.ClusterSpec({
        "worker": [
            "localhost:2220",
            "localhost:2221",
            "localhost:2222",
            "localhost:2223"
        ],
        "ps": [
            "localhost:3333"
        ]})

    job_name = FLAGS.job_name
    task_index = FLAGS.task_index
    server = tf.train.Server(cluster,
                             job_name=job_name,
                             task_index=task_index)
    if job_name == "ps":
        server.join()
    elif job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:" + str(task_index),
                cluster=cluster)):
            # Build model...
            sdfgan = SDFGAN(FLAGS,
                input_depth=FLAGS.input_depth,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_depth=FLAGS.output_depth,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                c_dim=FLAGS.c_dim,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                is_crop=FLAGS.is_crop,
                checkpoint_dir=FLAGS.checkpoint_dir,
                dataset_dir=FLAGS.dataset_dir,
                log_dir=FLAGS.log_dir,
                sample_dir=FLAGS.sample_dir)


            # The StopAtStepHook handles stopping after running given steps.
            hooks = [tf.train.StopAtStepHook(last_step=1000000)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(task_index == 0),
                                               checkpoint_dir=os.path.join(FLAGS.checkpoint_dir, "worker_" + str(task_index)), hooks=hooks) as sess:

            show_all_variables()
            if FLAGS.is_train:
                sdfgan.train(FLAGS, sess)
            else:
                raise ValueError("Needs to be in training mode for parallel training branch.")
                # if not sdfgan.load(FLAGS.checkpoint_dir):
                #     raise Exception("[!] Train a model first, then run test mode")
                # create_samples(sess, sdfgan, FLAGS)




        # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
        #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
        #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
        #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
        #                 [dcgan.h4_w, dcgan.h4_b, None])

        # Below is codes for visualization
        # OPTION = 1
        # visualize(sess, sdfgan, FLAGS, OPTION)


if __name__ == '__main__':
    tf.app.run()
