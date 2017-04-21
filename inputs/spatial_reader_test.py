from __future__ import division
from __future__ import print_function
import os
import cv2
import numpy as np
import random
from glob import glob
import tensorflow as tf

from config import config_agent
from config.config_agent import FLAGS, VARS
from kits import average_gradients
from solver import Solver
import inputs.spatial_reader as reader


def indexOfVideo(v, sub_v, num_frm):
    # find offset of timestep
    for t in range(v.shape[0]):
        clip = v[t:t + num_frm, :]
        step, height, width, channel = clip.shape
        # [step, height, width, channel] -> [height, width, channel * step]
        clip = np.transpose(clip, [1, 2, 0, 3]).reshape(
            height, width, step * channel)
        if np.array_equal(clip, sub_v):
            return t
            return -1


def indexofImage(npy, sub_npy):
    for y in range(npy.shape[0]):
        for x in range(npy.shape[1]):
            if np.array_equal(npy[y:y + sub_npy.shape[0], x:x + sub_npy.shape[1], :], sub_npy):
                return y, x
    print('not found x, y')


class ReaderTest(tf.test.TestCase):

    def __init__(self, *args, **kwargs):
        super(ReaderTest, self).__init__(*args, **kwargs)

        self.num_frm = 3
        self.test_times = 5
        self.sess = tf.Session()
        self.frm_dir = (
            '/home/wsy/dataset/ucf_101/rgb_data/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01')

    def read_video(self):
        frm_path_lst = [frm_path for frm_path in glob(
            os.path.join(self.frm_dir, 'image_*.jpg'))]
        # frm_len = len(frm_path_lst)
        frm_path_lst.sort()
        v_npy = np.array([cv2.imread(frm_path) for frm_path in frm_path_lst])
        print('entire video shape: ', v_npy.shape)
        return v_npy

    def test_read_rgb_randomly(self):
        v_npy = self.read_video()

        for i in range(self.test_times):
            print('test round: %d' % i)
            fn_output = reader.read_rgb_randomly(self.frm_dir, self.num_frm)
            assert len(fn_output) == 1
            print('fn_output shape: ', fn_output[0].shape)
            idx = indexOfVideo(v_npy, fn_output[0], self.num_frm)
            print('offset %d' % idx)

    def test_read_rgb_for_test(self):
        v_npy = self.read_video()
        len_frm = 3
        num_frm = 1
        out_lst = reader.read_rgb_for_test(
            self.frm_dir, [3, 3, len_frm], num_frm)
        for frm in out_lst:
            # print('test frm offset', indexOfVideo(v_npy, frm, len_frm // 3))
            if np.array_equal(frm, v_npy[0][:2, :2, :]):
                print('left-top')
            elif np.array_equal(frm, v_npy[0][:2, :2, :][:, ::-1, :]):
                print('left-top flip')
            elif np.array_equal(frm, v_npy[0][:2, -2:, :]):
                print('right-top')
            elif np.array_equal(frm, v_npy[0][:2, -2:, :][:, ::-1, :]):
                print('right-top flip')
            elif np.array_equal(frm, v_npy[0][-2:, :2, :]):
                print('left-bottom')
            elif np.array_equal(frm, v_npy[0][-2:, :2, :][:, ::-1, :]):
                print('left-bottom flip')
            elif np.array_equal(frm, v_npy[0][-3:, -3:, :]):
                print('right-bottom')
            elif np.array_equal(frm, v_npy[0][-3:, -3:, :][:, ::-1, :]):
                print('right-bottom flip')
            elif np.array_equal(frm, v_npy[0][127:129, 169:171, :]):
                print('mid')
            elif np.array_equal(frm, v_npy[0][127:129, 169:171, :][:, ::-1, :]):
                print('mid flip')

    def test_num_examples(self):
        # train or valid
        pass
        # test

if __name__ == '__main__':
    tf.test.main()
