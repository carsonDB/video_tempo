"""read batch of clips
Several threads read clips from videos
Parts:
    * read_thread: fetch clips and enqueue
    * seq_read_thread: fetch sequence of clips and enqueue
    * build_start_nodes: create placeholders
    * threads_ready: launch threads if necessary
types:
    * video: [depth, height, width, in_channels]
    * seq_video: [steps, depth, height, width, in_channels]
"""
import os
import math
import cv2
import json
import threading
import tensorflow as tf

# import local file
from config.config_agent import FLAGS


def read_wrapper(func):
    def inner(raw_input_uint8, label_input,
              sess, enqueue_op, coord):
        # read a list from file
        lst_path = FLAGS['lst_path']
        lst_path = os.path.expanduser(lst_path)
        with open(lst_path, 'r') as f:
            example_lst = json.load(f)
        pass
    return inner


def read_thread(raw_input_uint8, label_input,
                sess, enqueue_op, coord):
    """Reads examples from video data files and enqueue.
    """

    INPUT = FLAGS['input']
    clip_length = INPUT['clip_length']
    lst_path = FLAGS['lst_path']
    lst_path = os.path.expanduser(lst_path)
    # lock = threading.Lock()

    with open(lst_path, 'r') as f:
        example_lst = json.load(f)

    while not coord.should_stop():
        # loop until train(eval) ends
        for file_path, label_id in example_lst:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError("can't open: %s" % file_path)

            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            # for the safe length ...
            frame_count -= 5

            clip_count = int(math.floor(frame_count/clip_length))
            for i in range(clip_count):
                # each video produce several clips
                clip = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, i*clip_length)
                for j in range(clip_length):
                    # generate a list of images as a clip
                    ret, frm = cap.read()
                    if not ret:
                        raise ValueError("""
                          No.%d frame not available,
                          at file: %s
                          whose length is: %d
                          """ % (i*clip_length+j+1, file_path, frame_count))

                    # read as a specific resolution (width, height)
                    frm = cv2.resize(frm,
                                     tuple(reversed(INPUT['read_resolution'])))
                    # frame format: [height, width, in_channels]
                    clip.append(frm)

                # clip format: [depth, height, width, in_channels]
                if coord.should_stop():
                    cap.release()
                    return
                # with lock:
                sess.run(enqueue_op, feed_dict={raw_input_uint8: clip,
                                                label_input: label_id})
            cap.release()


def seq_read_thread(raw_input_uint8, label_input,
                    sess, enqueue_op, coord):
    """Reads sequence of examples from video data files and enqueue.
    """

    lst_path = FLAGS['lst_path']
    INPUT = FLAGS['input']
    clip_length = INPUT['clip_length']
    num_step = INPUT['num_step']

    with open(lst_path, 'r') as f:
        example_lst = json.load(f)

    while not coord.should_stop():
        # loop until train(eval) ends
        for file_path, label_id in example_lst:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError("can't open: %s" % file_path)

            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            # for the safe length ...
            frame_count -= 5

            clip_count = int(math.floor(frame_count/clip_length))
            if clip_count < num_step:
                print('video is not long enough: %s' % file_path)
                cap.release()
                continue
            # extract a sequence of clips
            seq_clip = []
            for i in range(num_step):
                clip = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, i*clip_length)
                for j in range(clip_length):
                    # generate a list of images as a clip
                    ret, frm = cap.read()
                    if not ret:
                        raise ValueError("""
                          No.%d frame not available,
                          at file: %s
                          whose length is: %d
                          """ % (i*clip_length+j+1, file_path, frame_count))

                    # read as a specific resolution (width, height)
                    frm = cv2.resize(frm,
                                     tuple(reversed(INPUT['read_resolution'])))
                    # frame format: [height, width, in_channels]
                    clip.append(frm)

                # clip format: [depth, height, width, in_channels]
                seq_clip.append(clip)

            if coord.should_stop():
                cap.release()
                return
            sess.run(enqueue_op, feed_dict={raw_input_uint8: seq_clip,
                                            label_input: [label_id]*num_step})
            cap.release()


def build_start_nodes():
    """
    start nodes:
        * raw_input_uint8
        * label_input
    return nodes:
        * raw_input_float32
        * label_input
    """
    INPUT = FLAGS['input']
    input_type = INPUT['type']
    clip_length = INPUT['clip_length']
    clip_size = ([clip_length] +
                 INPUT['read_resolution'] +
                 [INPUT['num_channel']])

    # raw input nodes
    if input_type == 'video':
        # shape: [depth, height, width, in_channels]
        label_input = tf.placeholder(tf.int32, shape=[])
        raw_input_uint8 = tf.placeholder(tf.uint8, shape=clip_size)
        raw_input_float32 = tf.cast(raw_input_uint8, tf.float32)

    elif input_type == 'seq_video':
        # shape: [step, depth, height, width, in_channels]
        num_step = INPUT['num_step']

        label_input = tf.placeholder(tf.int32, shape=[num_step])
        raw_input_uint8 = tf.placeholder(tf.uint8, shape=[num_step]+clip_size)
        raw_input_float32 = tf.cast(raw_input_uint8, tf.float32)

    # visualization clips
    # tf.image_summary('snapshot', raw_input_float32[0])

    return raw_input_float32, label_input, read_thread


def threads_ready():
    pass
