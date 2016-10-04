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
    reader_name = FLAGS['input_reader']

    sess = VARS['sess']
    coord = VARS['coord']
    if_test = VARS['if_test']

    # determine reader and preprocessor
    THIS['reader'] = reader = import_module('input.%s' % reader_name)
    preproc = import_module('input.video_proc')

    # build placeholders
    if if_test is True and reader.build_start_nodes_for_test:
        raw_input_ts, label_ts, mask, is_multi = \
            reader.build_start_nodes_for_test()
    else:
        raw_input_ts, label_ts, mask, is_multi = reader.build_start_nodes()

    # build preprocesss nodes
    #   one-example or multi-examples in
    input_ts, is_multi = preproc.proc(raw_input_ts, sess, group_head=is_multi)
    # outpus:
    #   one-example -> enqueue
    #   multi-examples -> enqueue_many

    mask_size = mask.get_shape().as_list()
    input_size = input_ts.get_shape().as_list()
    if is_multi:
        input_size = input_size[1:]

    # enqueue
    if QUEUE['type'] == 'shuffle':
        min_remain = QUEUE['min_remain']
        q = tf.RandomShuffleQueue(capacity, min_remain,
                                  [tf.float32, tf.int32, tf.float32],
                                  shapes=[input_size, [input_size[0]], mask_size])
    else:
        q = tf.FIFOQueue(capacity, [tf.float32, tf.int32, tf.float32],
                         shapes=[input_size, [input_size[0]], mask_size])
    VARS['queues'].append(q)
    # enqueue example
    if is_multi:
        raise ValueError('mask is not multi')
        THIS['enqueue_op'] = enqueue_op = \
            q.enqueue_many([input_ts, label_ts, mask])
    else:
        THIS['enqueue_op'] = enqueue_op = q.enqueue([input_ts, label_ts, mask])
    # dequeue batch
    input_batch, label_batch, mask_batch = q.dequeue_many(batch_size)

    # create reading threads if necessary
    VARS['threads'] += reader.create_threads(sess, enqueue_op, coord)

    return input_batch, label_batch, mask_batch


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
