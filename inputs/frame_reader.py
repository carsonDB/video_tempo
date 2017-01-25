"""frame_reader for clips
Raw_data format: every video consists of frames.
Output_data format:
    * train: a clip randomly cutted from frames(video).
    * other: sequence of clips cutted from frames(video).
"""
from __future__ import division
import os
import cv2
import json
import random
import numpy as np
import tensorflow as tf

from config.config_agent import FLAGS, VARS
from inputs.input_proto import Input_proto
from inputs import proc_kits as kits


def cut_out(frame_lst, read_resolution):
    # cut out into a list
    # Warning: OpenCv use BGR, not RGB
    re_lst = []
    for frm_path in frame_lst:
        if not os.path.exists(frm_path):
            raise ValueError('frame not exists: %s', frm_path)
        frm = cv2.imread(frm_path)
        #  resize if need
        if frm.shape[0:2] != read_resolution:
            frm = cv2.resize(frm, tuple(reversed(read_resolution)))
        re_lst.append(frm)

    return re_lst


def read_rgb(frm_dir, example_size, interval=None):
    """Two kinds of output
        * randomly cut out a clip from one video
        * cut out clips from a video sequatially
    """
    # example shape: [depth, height, width, channel]
    clip_len = example_size[0]

    frame_lst = [os.path.join(frm_dir, f)
                 for f in os.listdir(frm_dir)
                 if os.path.isfile(os.path.join(frm_dir, f))]
    # sort alphabet sequence
    frame_lst.sort()

    len_frame = len(frame_lst)
    # when video shorter than clip
    if len_frame < clip_len:
        last_frame = frame_lst[-1]
        frame_lst += [last_frame] * (clip_len - len_frame)

    if interval is not None:
        max_time_steps = FLAGS['input']['max_time_steps']
        # limit number of clips <= max_time_steps
        num_clip = min(max_time_steps,
                       (len_frame - clip_len)//interval + 1)

        start_id = 0
        out_lst = [cut_out(frame_lst[start_id + i*interval:
                                     start_id + i*interval + clip_len])
                   for i in range(num_clip)]
    else:
        start_id = random.randint(0, len(frame_lst) - clip_len)
        out_lst = cut_out(frame_lst[start_id:start_id + clip_len],
                          example_size[-3:-1])

    return np.array(out_lst)


class Reader(Input_proto):
    """read frames of videos

    """
    def __init__(self):
        super(Reader, self).__init__()
        self.lst_path = os.path.expanduser(FLAGS['lst_path'])

    def get_data(self):
        self.raw_inputs = {
            # data: [depth, height, width, channel]
            'data': tf.placeholder(tf.uint8, self.raw_size),
            # label: scalar
            'label': tf.placeholder(tf.int32, []),
        }

    def preproc(self):
        """preprocess data

        Returns:
            list: [processed data, label]
        """
        ts = self.raw_inputs['data']
        # clip: uint8 -> float32
        ts = tf.to_float(ts)
        # subtract mean: [depth, height, width, channel]
        ts = kits.subtract_mean(ts)
        # random or central crop: [height, width]
        crop_size = self.example_size[-3:-1]
        ts = (kits.random_crop(ts, crop_size)
              if self.mode == 'train' else
              kits.crop(ts, crop_size, 'mid'))

        if self.mode == 'train':
            # flip horizontally when train
            ts = kits.random_flip_left_right(ts, 0.5)

        return [ts, self.raw_inputs['label']]

    def read_thread(self):
        # read lst
        with open(self.lst_path, 'r') as f:
            example_lst = json.load(f)
        # shuffle
        if VARS['mode'] == 'train':
            random.shuffle(example_lst)

        # loop until train(eval) ends
        while not self.coord.should_stop():
            for frm_dir, label_id in example_lst:
                clip = read_rgb(frm_dir, self.raw_size)
                # enqueue
                self.sess.run(self.enqueue_op,
                              feed_dict={self.raw_inputs['data']: clip,
                                         self.raw_inputs['label']: label_id})
