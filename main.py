import os
import scipy.misc
import numpy as np

from model import SDFGAN
from utils import pp, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("d_learning_rate", 0.00005, "Learning rate of discrim for adam [0.00005]")
flags.DEFINE_float("g_learning_rate", 0.0025, "Learning rate of gen for adam [0.0025]")
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
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
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

    with tf.Session(config=run_config) as sess:

        sdfgan = SDFGAN(
            sess,
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
            sample_dir=FLAGS.sample_dir)

        show_all_variables()
        if FLAGS.is_train:
            sdfgan.train(FLAGS)
        else:
            if not sdfgan.load(FLAGS.checkpoint_dir):
                raise Exception("[!] Train a model first, then run test mode")

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
