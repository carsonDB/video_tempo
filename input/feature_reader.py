"""read one feature a time
Several threads read feature from hdf5-file
Parts:
    * read_thread: fetch clips and enqueue
    * build_start_nodes: create placeholders
    * build_start_nodes_for_test: create placeholders
    * is_custom: if need to launch threads manually
    * threads_ready: launch threads if necessary
types:
    feature: [max_time_steps, height, width, channels]
"""
from __future__ import division

import os
import h5py
import threading
import numpy as np
import tensorflow as tf

# import local file
from config.config_agent import FLAGS, VARS

# module variables
THIS = {}


def read_thread(feature, label, mask,
                sess, enqueue_op, coord):

    INPUT = FLAGS['input']
    num_step = INPUT['max_time_steps']
    feature_path = FLAGS['feature_path']
    feature_path = os.path.expanduser(feature_path)

    # read lst
    f = h5py.File(feature_path, 'r')
    feature_set = f['features']
    label_set = f['labels']
    if feature_set.shape[0] % num_step != 0:
        raise ValueError('your number of features must can be divided by %d',
                         num_step)
    num_example = feature_set.shape[0] // num_step

    # loop until train(eval) ends
    while not coord.should_stop():
        for i in range(num_example):
            feature_input = feature_set[i*num_step:(i + 1)*num_step]
            label_input = label_set[i*num_step:(i + 1)*num_step].astype(np.int)

            # enqueue
            sess.run(enqueue_op, feed_dict={feature: feature_input,
                                            label: label_input,
                                            mask: [1.0] * num_step})


def build_start_nodes():
    """
    start nodes:
        * raw_input_uint8
        * label_input
    return nodes:
        * raw_input_float32
        * label_input
    """
    # global variables declare
    INPUT = FLAGS['input']
    input_type = INPUT['type']
    num_step = INPUT['max_time_steps']
    example_size = INPUT['example_size']

    # raw input nodes
    if input_type == 'seq_feature':
        # shape: [max_time_steps, height, width, in_channels]
        label = tf.placeholder(tf.int32, shape=[num_step])
        feature = tf.placeholder(tf.float32,
                                 shape=example_size)
        mask = tf.placeholder(tf.float32, shape=[num_step])

    THIS['feature'] = feature
    THIS['label'] = label
    THIS['mask'] = mask

    return feature, label, mask, False


def build_start_nodes_for_test():
    """
    start nodes:
        * raw_input_uint8
        * label_input
    return nodes:
        * raw_input_float32
        * label_input
    """
    return build_start_nodes()


# if need to launch threads manually
def is_custom():
    return True


def create_threads(sess, enqueue_op, coord):

    QUEUE = FLAGS['input_queue']
    num_reader = QUEUE['num_reader']

    feature = THIS['feature']
    label = THIS['label']
    mask = THIS['mask']

    return [threading.Thread(target=read_thread,
                             args=(feature, label, mask,
                                   sess, enqueue_op, coord))
            for i in range(num_reader)
            ]


def start_threads(lst):
    for t in lst:
        t.start()
