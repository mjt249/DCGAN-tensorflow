from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from tqdm import tqdm
from ops import *
from utils import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class SDFGAN(object):
    def __init__(self, sess, input_depth=64, input_height=64, input_width=64, is_crop=True,
                 batch_size=64, sample_num=64,
                 output_depth=64, output_height=64, output_width=64, z_dim=200, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=1, dataset_name='shapenet',
                 input_fname_pattern='*.npy', checkpoint_dir=None, dataset_dir=None, log_dir=None, sample_dir=None,
                 classifier_epoch=25, seed=0):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width

        self.output_depth = output_depth
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim
        self.seed = seed
        np.random.seed(seed=seed)

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.dataset_dir = dataset_dir
        self.log_dir = log_dir
        self.sample_dir = sample_dir
        self.classifier_epoch = classifier_epoch
        self.build_model()

    def build_model(self):

        if self.is_crop:
            image_dims = [self.output_depth, self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.output_depth, self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.sample_inputs = tf.placeholder(
            tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

        inputs = self.inputs
        sample_inputs = self.sample_inputs

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(inputs)

        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G[:, 32, :, :])

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.d_accu_real = tf.reduce_sum(tf.cast(self.D > .5, tf.int32)) / self.D.get_shape()[0]
        self.d_accu_fake = tf.reduce_sum(tf.cast(self.D_ < .5, tf.int32)) / self.D_.get_shape()[0]
        self.d_accu = (self.d_accu_real + self.d_accu_fake) / 2

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train SFDGAN"""
        data = glob(os.path.join(self.dataset_dir, config.dataset, self.input_fname_pattern))
        np.random.shuffle(data)

        d_optim = tf.train.AdamOptimizer(config.d_learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.g_learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter(self.log_dir, self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        sample_files = data[0:self.sample_num]
        sample = [np.load(sample_file)[0, :, :, :] for sample_file in sample_files]
        if (self.is_grayscale):
            # sample_inputs = np.array(sample).astype(np.float32)[:, :, :, :, None]
            sample_inputs = np.array(sample).astype(np.float32)
        else:
            sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        d_accu_last_batch = .5
        for epoch in xrange(config.epoch):
            data = glob(os.path.join(
                self.dataset_dir, config.dataset, self.input_fname_pattern))
            batch_idxs = min(len(data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch = [
                    np.load(batch_file)[0, :, :, :] for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)[:, :, :, :, None]

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

                # Update D network if accuracy in last batch <= 80%
                if d_accu_last_batch < .8:
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={self.inputs: batch_images, self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update last batch accuracy
                d_accu_last_batch = self.sess.run([self.d_accu],
                                                  feed_dict={self.inputs: batch_images, self.z: batch_z})
                d_accu_last_batch = d_accu_last_batch[0]

                # # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                # _, summary_str = self.sess.run([g_optim, self.g_sum],
                #                                feed_dict={self.z: batch_z})
                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, d_accu: %.4f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG, d_accu_last_batch))

                if np.mod(counter, 200) == 1:
                    try:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                            },
                        )
                        np.save(self.sample_dir+'/train_{:02d}_{:04d}.npy'.format(config.sample_dir, epoch, idx), samples)
                        print("[Sample] d_loss: %.8f, g_loss: %.8f, d_accu: %.4f" % (d_loss, g_loss, d_accu_last_batch))
                    except:
                        print("Error when saving samples.")

                if np.mod(counter, 200) == 2:
                    self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv3d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv3d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv3d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv3d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, name='d_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            s_d, s_h, s_w = self.output_depth, self.output_height, self.output_width
            s_d2, s_h2, s_w2 = conv_out_size_same(s_d, 2), conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_d4, s_h4, s_w4 = conv_out_size_same(s_d2, 2), conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_d8, s_h8, s_w8 = conv_out_size_same(s_d4, 2), conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_d16, s_h16, s_w16 = conv_out_size_same(s_d8, 2), conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, self.gf_dim * 8 * s_d16 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_d16, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv3d(
                h0, [self.batch_size, s_d8, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv3d(
                h1, [self.batch_size, s_d4, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv3d(
                h2, [self.batch_size, s_d2, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv3d(
                h3, [self.batch_size, s_d, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def sampler(self, z):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_d, s_h, s_w = self.output_depth, self.output_height, self.output_width
            s_d2, s_h2, s_w2 = conv_out_size_same(s_d, 2), conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_d4, s_h4, s_w4 = conv_out_size_same(s_d2, 2), conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_d8, s_h8, s_w8 = conv_out_size_same(s_d4, 2), conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_d16, s_h16, s_w16 = conv_out_size_same(s_d8, 2), conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            h0 = tf.reshape(
                linear(z, self.gf_dim * 8 * s_d16 * s_h16 * s_w16, 'g_h0_lin'),
                [-1, s_d16, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv3d(h0, [self.batch_size, s_d8, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv3d(h1, [self.batch_size, s_d4, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv3d(h2, [self.batch_size, s_d2, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv3d(h3, [self.batch_size, s_d, s_h, s_w, self.c_dim], name='g_h4')

            return tf.nn.tanh(h4)
        
# Modules for classifier
    def build_classifier(self, config):
        if self.is_crop:
            image_dims = [self.output_depth, self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.output_depth, self.input_height, self.input_width, self.c_dim]

        self.n_class = int(config.classification_dataset[-2:])
        self.ct_inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='classifier_train_inputs')
        self.ct_labels = tf.placeholder(tf.int64, [self.batch_size], name='classifier_train_labels')
        self.ct_labels_onehot = tf.one_hot(self.ct_labels, depth=self.n_class)
        self.C, self.C_logits, _ = self.classifier(self.ct_inputs)

        def sigmoid_cross_entropy_with_logits(x, y):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            # try:
            #     return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            # except:
            #     return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.c_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.C_logits, self.ct_labels_onehot))
        self.c_loss_sum = scalar_summary("c_loss", self.c_loss)
        self.c_accu = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.C, axis=1), self.ct_labels), tf.int32)) \
                      / self.ct_labels.get_shape()[0]
        self.c_accu_sum = scalar_summary("c_accu", self.c_accu)

        t_vars = tf.trainable_variables()
        self.c_vars = [var for var in t_vars if 'c_' in var.name]

    def classifier(self, image):
        with tf.variable_scope("discriminator") as scope:
            scope.reuse_variables()

            h0 = lrelu(conv3d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv3d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv3d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv3d(h2, self.df_dim * 8, name='d_h3_conv')))
        with tf.variable_scope("classifier"):
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), self.n_class, name='c_h3_lin')
            
            return tf.nn.sigmoid(h4), h4, tf.reshape(h3, [self.batch_size, -1])
    
    def train_classifier(self, config):
        """Train classifier for classification dataset (ModelNet10/40) based on discriminator features"""
        train_files, train_labels, test_files, test_labels, _ = self.read_data(config)
        # define optimizing step
        c_optim = tf.train.AdamOptimizer(config.c_learning_rate, beta1=config.beta1) \
            .minimize(self.c_loss, var_list=self.c_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.c_sum = merge_summary([self.c_loss_sum, self.c_accu_sum])
        self.writer = SummaryWriter(self.log_dir, self.sess.graph)

        c_counter = 1
        could_load, counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        batch_idxs = min(len(train_files), config.train_size) // config.batch_size
        total_steps = config.classifier_epoch * batch_idxs
        prev_time = -np.inf
        for c_epoch in xrange(config.classifier_epoch):
            for idx in xrange(0, batch_idxs):
                batch_files = train_files[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_labels = train_labels[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch = [
                    np.load(batch_file)[0, :, :, :] for batch_file in batch_files]
                batch_inputs = np.array(batch).astype(np.float32)[:, :, :, :, None]

                # Update fully connected layer for classification
                _, summary_str = self.sess.run([c_optim, self.c_sum],
                                               feed_dict={self.ct_inputs: batch_inputs,
                                                          self.ct_labels: batch_labels})
                self.writer.add_summary(summary_str, c_counter)
                c_loss, mini_batch_accu = self.sess.run([self.c_loss, self.c_accu],
                                                        feed_dict={self.ct_inputs: batch_inputs,
                                                                   self.ct_labels: batch_labels})
                now_time = time.time()
                time_per_iter = now_time - prev_time
                prev_time = now_time
                eta = (total_steps - c_counter) * time_per_iter
                c_counter += 1
                counter += 1

                print("Epoch: [%2d] [%4d/%4d] sec/iter: %4.4f, eta: %4.4f mins, c_loss: %.8f, minibatch_accu: %.8f"
                      % (c_epoch, idx, batch_idxs, time_per_iter, eta/60, c_loss, mini_batch_accu))

                # evaluate and print train/test set error on small sample
                if np.mod(c_counter, 500) == 1:
                    print("Evaluating training set accuracy:")
                    train_accu = self.eval_classifier(config, partition='train')
                    print("Evaluating test set accuracy:")
                    test_accu = self.eval_classifier(config, partition='test')
                    print("Accuracy: train_accu %.8f, test_accu %.8f." % (train_accu, test_accu))

        # print final accuracy
        train_accu = self.eval_classifier(config, partition='train')
        test_accu = self.eval_classifier(config, partition='test')
        print("FINAL Accuracy: train_accu %.8f, test_accu %.8f." % (train_accu, test_accu))
        self.save(config.checkpoint_dir, counter)

    def eval_classifier(self, config, partition='test'):
        """Evaluate classifier accuracy given inputs and labels."""
        if partition == 'test':
            _, _, files, labels, _ = self.read_data(config)
        else:
            files, labels, _, _, _ = self.read_data(config)

        batch_idxs = len(files) // config.batch_size
        correct, total = 0, 0
        for idx in tqdm(xrange(0, batch_idxs)):
            batch_files = files[idx * config.batch_size:(idx + 1) * config.batch_size]
            batch_labels = labels[idx * config.batch_size:(idx + 1) * config.batch_size]
            batch = [
                np.load(batch_file)[0, :, :, :] for batch_file in batch_files]
            batch_inputs = np.array(batch).astype(np.float32)[:, :, :, :, None]
            accu = self.c_accu.eval(feed_dict={self.ct_inputs: batch_inputs,
                                               self.ct_labels: batch_labels})
            correct += int(accu*batch_labels.shape[0])
            total += int(batch_labels.shape[0])
        final_accu = correct / total

        return final_accu


    def read_data(self, config):
        """Read data for classifier"""
        # read data
        if config.classification_dataset == 'ModelNet10' or config.classification_dataset == 'ModelNet40':
            data_classes = glob(os.path.join(self.dataset_dir, config.classification_dataset) + '/*/')
        else:
            raise Exception('only implemented classification for ModelNet10 and ModelNet40.')
        train_f = [glob(os.path.join(c, 'train', config.input_fname_pattern)) for c in data_classes]
        test_f = [glob(os.path.join(c, 'test', config.input_fname_pattern)) for c in data_classes]
        train_files, train_labels, test_files, test_labels = [], [], [], []
        for num, t_f in enumerate(train_f):
            train_files += t_f
            train_labels += [num * np.ones([len(t_f)], dtype=np.int64)]
        train_labels = np.concatenate(train_labels, axis=0)
        for num, t_f in enumerate(test_f):
            test_files += t_f
            test_labels += [num * np.ones([len(t_f)], dtype=np.int64)]
        test_labels = np.concatenate(test_labels, axis=0)
        label_names = [os.path.basename(c[:-1]) for c in data_classes]

        # shuffle training data
        rand_seq = np.random.permutation(len(train_files))
        train_files = [train_files[ind] for ind in rand_seq]
        train_labels = train_labels[rand_seq]

        return train_files, train_labels, test_files, test_labels, label_names

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_depth, self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "SDFGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
