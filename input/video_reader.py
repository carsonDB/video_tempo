"""read batch of clips
Several threads read clips from videos
Parts:
    * read_thread: fetch clips and enqueue
    * seq_read_thread: fetch sequence of clips and enqueue
    * build_start_nodes: create placeholders
    * is_custom: if need to launch threads manually
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
import random
import numpy as np
import tensorflow as tf

# import local file
from config.config_agent import FLAGS

# global variables
RAW_INPUT_FLOAT32 = None
LABEL_INPUT = None


def read_rgb(frm_dir, length):
    """
    random cut out a clip from one video
    """
    read_resolution = FLAGS['input']['read_resolution']

    frame_lst = [os.path.join(frm_dir, f)
                 for f in os.listdir(frm_dir)
                 if os.path.isfile(os.path.join(frm_dir, f))]
    # sort alphabet sequence
    frame_lst.sort()

    len_frame = len(frame_lst)
    if len_frame < length:
        last_frame = frame_lst[-1]
        frame_lst += [last_frame] * (length - len_frame)

    start_id = random.randint(0, len(frame_lst) - length)

    # cut out
    re_lst = []
    for frm_path in frame_lst[start_id:(start_id + length)]:
        if not os.path.exists(frm_path):
            raise ValueError('frame not exists: %s', frm_path)
        frm = cv2.imread(frm_path)
        #  resize if need
        if frm.shape[0:2] != read_resolution:
            frm = cv2.resize(frm, tuple(reversed(read_resolution)))
        re_lst.append(frm)

    return re_lst


def read_flow(frm_dir, length):
    """
    random cut out a clip from one video
    """
    read_resolution = FLAGS['input']['read_resolution']

    frame_lst = [os.path.join(frm_dir, f)
                 for f in os.listdir(frm_dir)
                 if os.path.isfile(os.path.join(frm_dir, f))]
    len_frame = len(frame_lst) / 2
    if len_frame < length:
        raise ValueError('video not long enough: %s', frm_dir)
    start_id = random.randint(0, len_frame - length)

    # cut out
    re_lst = []
    for frm_id in range(start_id+1, start_id+length+1):
        # x and y direction optical flows
        frm_x_path = os.path.join(frm_dir, 'flow_x_%04d.jpg' % frm_id)
        frm_y_path = os.path.join(frm_dir, 'flow_y_%04d.jpg' % frm_id)
        if ((not os.path.exists(frm_x_path))
                or (not os.path.exists(frm_y_path))):
            raise ValueError('flow %d not exists', frm_id)
        # read grayscale image
        frm_x = cv2.imread(frm_x_path, cv2.IMREAD_GRAYSCALE)
        frm_y = cv2.imread(frm_y_path, cv2.IMREAD_GRAYSCALE)
        #  resize if need
        if frm_x.shape[0:2] != read_resolution:
            frm_x = cv2.resize(frm_x, tuple(reversed(read_resolution)))
        if frm_y.shape[0:2] != read_resolution:
            frm_y = cv2.resize(frm_y, tuple(reversed(read_resolution)))

        re_lst.append(np.dstack((frm_x, frm_y)))

    return re_lst


def clip_read_thread(raw_input_uint8, label_input,
                     sess, enqueue_op, coord):

    INPUT = FLAGS['input']
    num_channel = INPUT['num_channel']
    clip_length = INPUT['clip_length']
    lst_path = FLAGS['lst_path']
    lst_path = os.path.expanduser(lst_path)

    # determin read rgb or optical_flow
    if num_channel == 3:
        read_clip = read_rgb
    elif num_channel == 2:
        read_clip = read_flow

    # read lst
    with open(lst_path, 'r') as f:
        example_lst = json.load(f)
    # shuffle
    random.shuffle(example_lst)

    # loop until train(eval) ends
    while not coord.should_stop():
        for frm_dir, label_id in example_lst:
            clip = read_clip(frm_dir, clip_length)
            # enqueue
            try:
                sess.run(enqueue_op, feed_dict={raw_input_uint8: clip,
                                                label_input: label_id})
            except:
                return


# deprecated !!!
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


# deprecated !!!
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
    # global variables declare
    global RAW_INPUT_FLOAT32
    global LABEL_INPUT

    INPUT = FLAGS['input']
    input_type = INPUT['type']
    clip_length = INPUT['clip_length']
    clip_size = ([clip_length] +
                 INPUT['read_resolution'] +
                 [INPUT['num_channel']])

    # raw input nodes
    if input_type == 'video':
        # shape: [depth, height, width, in_channels]
        LABEL_INPUT = tf.placeholder(tf.int32, shape=[])
        raw_input_uint8 = tf.placeholder(tf.uint8, shape=clip_size)
        RAW_INPUT_FLOAT32 = tf.cast(raw_input_uint8, tf.float32)

    elif input_type == 'seq_video':
        # shape: [step, depth, height, width, in_channels]
        num_step = INPUT['num_step']

        LABEL_INPUT = tf.placeholder(tf.int32, shape=[num_step])
        raw_input_uint8 = tf.placeholder(tf.uint8, shape=[num_step]+clip_size)
        RAW_INPUT_FLOAT32 = tf.cast(raw_input_uint8, tf.float32)

    # visualization clips
    # tf.image_summary('snapshot', raw_input_float32[0])

    return RAW_INPUT_FLOAT32, LABEL_INPUT


# if need to launch threads manually
def is_custom():
    return True


def create_threads(sess, enqueue_op, coord):

    QUEUE = FLAGS['input_queue']
    num_reader = QUEUE['num_reader']

    return [threading.Thread(target=clip_read_thread,
                             args=(RAW_INPUT_FLOAT32, LABEL_INPUT,
                                   sess, enqueue_op, coord))
            for i in range(num_reader)
            ]


def start_threads(lst):
    for t in lst:
        t.start()


# def pause_threads(lst):
#     for t in lst:
#         t.


# def main():
#     frm_path = '/home/user/data/ucf101_rgb_img/Typing/v_Typing_g21_c02/'
#     re_lst = read_rgb(frm_path, 3, 10)
#     print re_lst[0]


# if __name__ == '__main__':
#     main()
