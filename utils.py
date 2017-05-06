"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import os
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
from tqdm import tqdm

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(
        x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def inverse_transform(images):
    return (images + 1.) / 2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append(
                        {"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2 ** (int(layer_idx) + 2), 2 ** (int(layer_idx) + 2),
                   W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'", "").split()))


def create_samples(sess, sdfgan, config):
    z_sample = np.random.uniform(-1, 1, size=(config.batch_size,sdfgan.z_dim))
    samples = sess.run(sdfgan.sampler, feed_dict={sdfgan.z: z_sample})
    fname = os.path.join(config.sample_dir, "samples.npy")
    np.save(fname, samples)


def convert_latent_vector(sdfgan, config):
    sdfgan.n_class = config.classification_dataset[-2:]
    train_files, train_labels, test_files, test_labels, label_names = sdfgan.read_data(config)

    # build graph
    if sdfgan.is_crop:
        image_dims = [sdfgan.output_depth, sdfgan.output_height, sdfgan.output_width, sdfgan.c_dim]
    else:
        image_dims = [sdfgan.output_depth, sdfgan.input_height, sdfgan.input_width, sdfgan.c_dim]
    ct_inputs = tf.placeholder(tf.float32, [config.batch_size] + image_dims, name='converter_train_inputs')
    _, _, latent_vect = sdfgan.classifier(ct_inputs)

    all_train_data, all_test_data = [], []

    # process train data
    batch_idxs = len(train_files) // config.batch_size
    print("processing train data:")
    for idx in tqdm(xrange(0, batch_idxs)):
        batch_files = train_files[idx * config.batch_size:(idx + 1) * config.batch_size]
        batch = [np.load(batch_file)[0, :, :, :] for batch_file in batch_files]
        batch_inputs = np.array(batch).astype(np.float32)[:, :, :, :, None]
        all_train_data.append(latent_vect.eval(feed_dict={ct_inputs: batch_inputs}))

    # process test data
    batch_idxs = len(test_files) // config.batch_size
    print("processing test data:")
    for idx in tqdm(xrange(0, batch_idxs)):
        batch_files = test_files[idx * config.batch_size:(idx + 1) * config.batch_size]
        batch = [np.load(batch_file)[0, :, :, :] for batch_file in batch_files]
        batch_inputs = np.array(batch).astype(np.float32)[:, :, :, :, None]
        all_test_data.append(latent_vect.eval(feed_dict={ct_inputs: batch_inputs}))

    all_train_data = np.concatenate(all_train_data, axis=0)
    all_test_data = np.concatenate(all_test_data, axis=0)

    # save processed data
    cond_mkdir("latent_vectors")
    np.save("latent_vectors/train_data_latent_vect.npy", all_train_data)
    np.save("latent_vectors/test_data_latent_vect.npy", all_test_data)
    np.save("latent_vectors/train_labels.npy", train_labels)
    np.save("latent_vectors/test_labels.npy", test_labels)
    target = open("latent_vectors/label_names.txt", 'w')
    for i, lname in enumerate(label_names):
        target.write("id: {0}, ".format(i) + lname + "\n")
    target.close()


def cond_mkdir(directory):
    """Conditionally make directory if it does not exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print('Directory ' + directory + ' existed. Did not create.')
