"""Handle all kinds of inputs
Take charge of input process:
    * readers (with multi threads)
    * preprocess
Type of inputs:
    * image
    * video
    * sequence of clips
"""
from importlib import import_module
import tensorflow as tf

# import local files
from config.config_agent import FLAGS


def read(sess, coord):

    QUEUE = FLAGS['input_queue']
    capacity = QUEUE['capacity']
    batch_size = FLAGS['batch_size']

    # determine reader and preprocessor
    reader = import_module('input.video_reader')
    preproc = import_module('input.video_proc')

    # build placeholders
    raw_input_ts, label_ts = reader.build_start_nodes()

    # build preprocesss nodes
        # get list of tensors
    input_ts = preproc.proc(raw_input_ts, sess)

    input_size = input_ts.get_shape()
    # enqueue
    if QUEUE['type'] == 'shuffle':
        min_remain = QUEUE['min_remain']
        q = tf.RandomShuffleQueue(capacity, min_remain,
                                  [tf.float32, tf.int32],
                                  shapes=[input_size, []])
    else:
        q = tf.FIFOQueue(capacity, [tf.float32, tf.int32],
                         shapes=[input_size, []])

    enqueue_op = q.enqueue([input_ts, label_ts])
    # dequeue
    input_batch, label_batch = q.dequeue_many(batch_size)

    # Start threads to enqueue data asynchronously, and hide I/O latency.
    thread_lst = reader.threads_ready(sess, enqueue_op, coord)

    return input_batch, label_batch, thread_lst
