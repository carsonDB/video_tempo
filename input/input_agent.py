"""Handle all kinds of inputs
Take charge of input process:
    * readers (with multi threads)
    * preprocess
Type of inputs:
    * image
    * video
    * sequence of clips

Components:
    * read
    * close
"""
from importlib import import_module
import tensorflow as tf

# import local files
from config.config_agent import FLAGS, VARS

# module FLAGS
THIS = {}


def read():

    QUEUE = FLAGS['input_queue']
    capacity = QUEUE['capacity']
    batch_size = FLAGS['batch_size']
    sess = VARS['sess']

    # determine reader and preprocessor
    THIS['reader'] = reader = import_module('input.video_reader')
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
    THIS['queue'] = q

    THIS['enqueue_op'] = q.enqueue([input_ts, label_ts])
    # dequeue
    input_batch, label_batch = q.dequeue_many(batch_size)

    return input_batch, label_batch


def launch():

    sess = VARS['sess']
    coord = VARS['coord']
    reader = THIS['reader']
    enqueue_op = THIS['enqueue_op']

    # start threads in the collection of 'queue_runners'
    tf.start_queue_runners(sess=sess, coord=coord)
    # start threads self-defined
    if reader.is_custom:
        THIS['threads'] = reader.threads_ready(sess, enqueue_op, coord)


def close():

    sess = VARS['sess']
    coord = VARS['coord']
    threads = THIS['threads']
    queue = THIS['queue']

    # disable equeue op, in case of readers blocking
    sess.run(queue.close(cancel_pending_enqueues=True))
    coord.request_stop()
    coord.join(threads)
    sess.close()
