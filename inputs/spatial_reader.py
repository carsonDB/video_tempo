"""spatial_reader for frames
Raw_data format: every video consists of frames.
Output_data format:
    * train: a frame randomly cutted from frames(video).
"""
from __future__ import division
import os
import cv2
import json
import numpy as np
import random
import tensorflow as tf

from config.config_agent import FLAGS, VARS
from inputs.input_proto import Input_proto
from inputs import proc_kits as kits


def read_rgb_for_train(frm_dir):
    """ train mode: random_crop to get a frame
        output: [a frame]
    """
    frame_lst = [os.path.join(frm_dir, f)
                 for f in os.listdir(frm_dir)
                 if os.path.isfile(os.path.join(frm_dir, f))]
    # sort alphabet sequence
    frame_lst.sort()
    len_video = len(frame_lst)
    re_lst = []

    # random select a frame from a video
    start_id = random.randint(0, len_video - 1)
    frm_path = frame_lst[start_id]
    if not os.path.exists(frm_path):
        raise ValueError('frame not exists: %s', frm_path)
    frm = cv2.imread(frm_path)
    re_lst.append(frm)

    return re_lst


def _corner_crops(npy, re_size):
    """Only crop along [height, width]

    Args:
        npy (numpy): [..., height, width, in_channels]
        size (list): [height, width]

    Returns:
        list of npy: cropped npy with reshaped [re_height, re_width]
    """
    raw_shape = list(npy.shape)
    height, width = raw_shape[-3:-1]
    re_height, re_width = re_size

    # only allow 5 kinds of crops
    H_W_starts = [
        # mid crop
        [(height - re_height) // 2, (width - re_width) // 2],
        # left_top
        [0, 0],
        # left_bottom
        [height - re_height, 0],
        # right_top
        [0, width - re_width],
        # right_bottom
        [height - re_height, width - re_width]
    ]

    # reshape to [..., height, width, in_channels]
    npy = np.reshape(npy, [-1] + raw_shape[-3:])

    raw_shape[-3:-1] = re_size
    re_lst = [np.reshape(npy[:, H_start:H_start + re_height,
                             W_start:W_start + re_width, :],
                         raw_shape)
              for H_start, W_start in H_W_starts]

    return re_lst


def _double_flip(npy):
    """double flip horizontally
    output: list of npy with twice the number
    """
    # reshape to [..., width, in_channels]
    raw_shape = list(npy.shape)

    mirror_npy = np.reshape(npy, [-1] + raw_shape[-2:])
    mirror_npy = np.reshape(mirror_npy[:, ::-1, :], raw_shape)

    return [npy, mirror_npy]


def read_rgb_for_test(frm_dir, example_size, num_frm):
    """ eval mode: corner crops and double flip, output 10 patches
            from a frame.
        output: a list of num_frm * 10 patches.
    """
    frame_lst = [os.path.join(frm_dir, f)
                 for f in os.listdir(frm_dir)
                 if os.path.isfile(os.path.join(frm_dir, f))]
    # sort alphabet sequence
    frame_lst.sort()
    len_video = len(frame_lst)
    frm_lst = []
    # select fix frames from video with equal space
    len_space = len_video // num_frm

    for i in range(num_frm):
        frm_path = frame_lst[i * len_space]
        if not os.path.exists(frm_path):
            raise ValueError('frame not exists: %s', frm_path)
        frm = cv2.imread(frm_path)
        frm_lst.append(frm)

    # corner crops
    cropped_lst = []
    for frm in frm_lst:
        cropped_lst += _corner_crops(frm, example_size[-3:-1])
    # double flip
    re_lst = []
    for frm in cropped_lst:
        re_lst += _double_flip(frm)

    return re_lst


class Reader(Input_proto):
    """read frames of videos
    """

    def __init__(self):
        super(Reader, self).__init__()
        self.raw_size = self.INPUT['raw_size']
        self.num_per_video = (FLAGS['num_per_video']
                              if 'num_per_video' in FLAGS
                              else None)
        self.data_path = os.path.expanduser(FLAGS['data_path'])

    def get_data(self):
        self.raw_inputs = {
            # X: [depth, height, width, channel]
            'X': (tf.placeholder(tf.uint8, self.raw_size)
                  if self.mode in ['train', 'eval'] else
                  tf.placeholder(tf.uint8, self.example_size)),
            # Y: scalar
            'Y': tf.placeholder(tf.int32, []),
            # name(frame path): scalar
            'name': tf.placeholder(tf.string, [])
        }

        raw_X = self.raw_inputs['X']
        # clip: uint8 -> float32
        ts = tf.to_float(raw_X)
        # random or central crop: [height, width]
        crop_size = self.example_size[-3:-1]
        if self.mode == 'train':
            ts = kits.random_crop(ts, crop_size)
        elif self.mode == 'eval':
            ts = kits.crop(ts, crop_size, 'mid')

        # flip horizontally when train
        if self.mode == 'train':
            ts = kits.random_flip_left_right(ts, 0.5)

        # subtract mean: [..., height, width, channel]
        ts = kits.subtract_mean(ts)

        return {'X': ts, 'Y': self.raw_inputs['Y'],
                'name': self.raw_inputs['name']}

    def read_thread(self):
        # read lst
        with open(self.data_path, 'r') as f:
            example_lst = json.load(f)
        # shuffle
        if VARS['mode'] in ['train', 'eval']:
            random.shuffle(example_lst)

        # loop until train(eval) ends
        while not self.coord.should_stop():
            for frm_dir, label_id in example_lst:
                clips = (read_rgb_for_train(frm_dir)
                         if self.mode in ['train', 'eval'] else
                         read_rgb_for_test(frm_dir, self.example_size,
                                           self.num_per_video))
                for c in clips:
                    # enqueue
                    self.sess.run(self.enqueue_op,
                                  feed_dict={self.raw_inputs['X']: c,
                                             self.raw_inputs['Y']: label_id,
                                             self.raw_inputs['name']: frm_dir})

    def num_examples(self):
        # read lst
        with open(self.data_path, 'r') as f:
            example_lst = json.load(f)

        if self.mode in ['train', 'eval']:
            # one frame a video
            return len(example_lst)
        else:
            # eval: num_per_video * 10 * num_videos
            return self.num_per_video * 10 * len(example_lst)

if __name__ == '__main__':
    # use for test codes
    frm = cv2.imread('../data/two_stream/image_0001.jpg')
    p_lst = read_rgb_for_test('/home/wsy/dataset/ucf_101/'
                              'rgb_data/ApplyEyeMakeup/'
                              'v_ApplyEyeMakeup_g01_c01',
                              [224, 224, 3], 1)
    for p in p_lst:
        for y in range(256):
            for x in range(340):
                if np.array_equal(frm[y:y + 224, x:x + 224], p):
                    print(y, x, 'no')
                elif np.array_equal(frm[y:y + 224, x:x + 224], p[:, ::-1, :]):
                    print(y, x, 'mirror')
