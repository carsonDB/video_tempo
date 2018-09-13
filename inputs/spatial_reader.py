"""spatial_reader for frames (custom python reader)
Raw_data format: every video consists of frames.
Output_data format:
    * train: frames randomly cutted from frames(video).
"""
from __future__ import division
from __future__ import print_function
import os
from glob import glob
import threading
import json
import random
import cv2
import sys
import json
import numpy as np
import traceback
import random
import tensorflow as tf

from config.config_agent import FLAGS, VARS
from inputs.input_proto import Input_proto
from inputs import video_kits as kits
from kits import pp


def read_rgb_randomly(frm_dir, raw_size=None, num=1):
    """randomly slice from a video(consists of frames) with continuous frames

    Args:
        frm_dir (string): absPath of video.
        num (int): number of sliced frames (continuous).

    Returns:
        list of frames [height, width, channel(3)].
    """
    frame_lst = [frm_path for frm_path in glob(
        os.path.join(frm_dir, 'image_*.jpg'))]
    # sort alphabet sequence
    frame_lst.sort()
    len_video = len(frame_lst)
    re_lst = []

    # random select a frame from a video
    start_id = random.randint(0, len_video - num)
    frm_lst = []
    for idx in range(start_id, start_id + num):
        frm_path = frame_lst[idx]
        if not os.path.exists(frm_path):
            raise ValueError('frame not exists: %s', frm_path)
        frm = cv2.imread(frm_path)
        if raw_size and frm.shape != raw_size:
            height, width, channel = raw_size
            frm = cv2.resize(frm, (width, height))
        frm_lst.append(frm)

    clip = np.concatenate(frm_lst, axis=-1)
    re_lst.append(clip)
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
            from a frame(or sequence frames).
        output: a list of num_frm * 10 patches.
    """
    frame_lst = [frm_path for frm_path in glob(
        os.path.join(frm_dir, 'image_*.jpg'))]
    # sort alphabet sequence
    frame_lst.sort()
    len_video = len(frame_lst)
    len_frm = example_size[-1] // 3
    frm_lst = []

    # select fix frames from video with equal space
    margin = (len_video - len_frm) // num_frm
    for i in range(num_frm):
        frm_path_lst = frame_lst[i * margin:i * margin + len_frm]
        if len(frm_path_lst) != len_frm:
            print('frm_path_lst != len_frm')
        sub_frm_lst = []
        for frm_path in frm_path_lst:
            if not os.path.exists(frm_path):
                raise ValueError('frame not exists: %s', frm_path)
            sub_frm_lst.append(cv2.imread(frm_path))
        frm_lst.append(np.concatenate(sub_frm_lst, axis=-1))

    # corner crops
    cropped_lst = []
    for frm in frm_lst:
        cropped_lst += _corner_crops(frm, example_size[-3:-1])
    # double flip
    re_lst = []
    for frm in cropped_lst:
        re_lst += _double_flip(frm)

    return re_lst


def get_example_lst(path):
    # read lst (I/O)
    if not os.path.isfile(path):
        print('no such data list file: %s' % path)

    with open(path, 'r') as f:
        example_lst = json.load(f)
    return example_lst


class Reader(Input_proto):
    """read frames of videos
    """

    def __init__(self):
        super(Reader, self).__init__()
        self.raw_size = self.INPUT['raw_size']
        self.seq_len = self.INPUT['seq_len']
        self.num_per_video = (FLAGS['num_per_video']
                              if 'num_per_video' in FLAGS
                              else None)
        self.data_path = os.path.expanduser(FLAGS['data_path'])
        self.example_lst = get_example_lst(self.data_path)

    def get_data(self):
        self.raw_inputs = {
            # X: [depth, height, width, channel]
            'X': (tf.placeholder(tf.uint8, self.raw_size)
                  if self.mode in ['train', 'eval'] else
                  tf.placeholder(tf.uint8, self.example_size)),
            # Y: scalar
            'Y': tf.placeholder(tf.int32, []),
        }
        # name(frame path): scalar
        if VARS['mode'] == 'test':
            self.raw_inputs['name'] = tf.placeholder(tf.string, [])

        pp(scope='preprocess')
        raw_X = self.raw_inputs['X']
        # clip: uint8 -> float32
        ts = tf.to_float(raw_X)

        # subtract mean: [..., height, width, channel]
        ts = kits.subtract_mean(ts)

        # # pixels scaled to [0, 1] for convenience of color_jitter
        # ts = tf.image.convert_image_dtype(ts, dtype=tf.float32)

        # random or central crop: [height, width]
        crop_size = self.example_size[-3:-1]
        if self.mode == 'train':
            # ts = kits.random_size_and_crop(ts, self.example_size[:2])
            # ts = kits.color_jitter(ts)
            # ts = kits.random_crop(ts, self.example_size[:2])
            ts = kits.random_fix_crop_with_multi_scale(ts,
                                                       [256, 224, 192, 168],
                                                       crop_size)
            ts = kits.random_flip_left_right(ts, 0.5)

            # # [0, 1] -> [-1, 1]
            # ts = tf.subtract(ts, 0.5)
            # ts = tf.multiply(ts, 2.0)
            # pp('normalize [0, 1] -> [-1, 1]')
        elif self.mode == 'eval':
            ts = kits.crop(ts, crop_size, 'mid')
            print('eval: central crop')

        if VARS['mode'] in ['train', 'eval']:
            return {'X': ts, 'Y': self.raw_inputs['Y']}
        else:
            return {'X': ts, 'Y': self.raw_inputs['Y'],
                    'name': self.raw_inputs['name']}

    def create_threads(self):
        num_thread = self.QUEUE['num_thread']
        self.py_threads += [threading.Thread(target=self.py_thread)
                            for _ in range(num_thread)]

    def py_thread(self):
        # get the copy one of example_lst
        example_lst = self.example_lst[:]

        # loop until train(eval) ends
        while not self.coord.should_stop():
            # shuffle for every epoch
            if VARS['mode'] in ['train', 'eval']:
                random.shuffle(example_lst)

            for frm_dir, label_id in example_lst:

                clips = (read_rgb_randomly(frm_dir, raw_size=self.raw_size,
                                           num=self.seq_len)
                         if self.mode in ['train', 'eval'] else
                         read_rgb_for_test(frm_dir, self.example_size,
                                           self.num_per_video))
                for c in clips:
                    # enqueue
                    feed_dict = {self.raw_inputs['X']: c,
                                 self.raw_inputs['Y']: label_id, }
                    # for video_test
                    if VARS['mode'] == 'test':
                        feed_dict[self.raw_inputs['name']] = frm_dir
                    try:
                        self.sess.run(self.enqueue_op,
                                      feed_dict=feed_dict)
                    except:
                        # traceback.print_exc(file=sys.stdout)
                        return

    def num_examples(self):
        if self.mode in ['train', 'eval']:
            # one frame a video
            return len(self.example_lst)
        else:
            # eval: num_per_video * 10 * num_videos
            return self.num_per_video * 10 * len(self.example_lst)

if __name__ == '__main__':
    from inputs import spatial_reader_test, video_kits_test
    tf.test.main()
