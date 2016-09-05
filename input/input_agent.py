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
from config.config_agent import FLAGS

# module FLAGS
local_FLAGS = {}


def read():

    QUEUE = FLAGS['input_queue']
    capacity = QUEUE['capacity']
    batch_size = FLAGS['batch_size']
    sess = FLAGS['sess']
    coord = FLAGS['coord']

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
    local_FLAGS['queue'] = q

    enqueue_op = q.enqueue([input_ts, label_ts])
    # dequeue
    input_batch, label_batch = q.dequeue_many(batch_size)

    # Start threads to enqueue data asynchronously, and hide I/O latency.
    thread_lst = reader.threads_ready(sess, enqueue_op, coord)

    # local FLAGS
    local_FLAGS['readers'] = thread_lst

    return input_batch, label_batch


def close():

    sess = FLAGS['sess']
    coord = FLAGS['coord']
    readers = local_FLAGS['readers']
    queue = local_FLAGS['queue']

    # disable equeue op, in case of readers blocking
    sess.run(queue.close(cancel_pending_enqueues=True))
    coord.request_stop()
    coord.join(readers)
    sess.close()
