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
    coord = VARS['coord']

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
    VARS['queues'].append(q)
    # enqueue example
    THIS['enqueue_op'] = enqueue_op = q.enqueue([input_ts, label_ts])
    # dequeue batch
    input_batch, label_batch = q.dequeue_many(batch_size)

    # create reading threads if necessary
    VARS['threads'] += reader.create_threads(sess, enqueue_op, coord)

    return input_batch, label_batch


def launch():
    reader = THIS['reader']
    reader.start_threads(VARS['threads'])


# def pause():
#     reader = THIS['reader']
#     reader.pause_threads(VARS['threads'])


def close():

    sess = VARS['sess']
    coord = VARS['coord']
    queues = VARS['queues']

    # disable equeue op, in case of readers blocking
    coord.request_stop()
    for q in queues:
        sess.run(q.close(cancel_pending_enqueues=True))
    threads = VARS['threads']
    coord.join(threads)
    sess.close()
