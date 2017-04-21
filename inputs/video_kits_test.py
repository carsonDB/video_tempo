from __future__ import division
import cv2
import numpy as np
import tensorflow as tf

from config import config_agent
from config.config_agent import FLAGS, VARS
from kits import average_gradients
from solver import Solver
import inputs.video_kits as kits


class KitsTest(tf.test.TestCase):

    def __init__(self, *args, **kwargs):
        super(KitsTest, self).__init__(*args, **kwargs)
        self.sess = tf.Session()
        self.input_shape = [256, 340, 3]
        self.input = tf.constant(
            cv2.imread('./data/two_stream/image_0001.jpg'))
        self.new_size = [224, 224]

    def indexof(self, npy, sub_npy):
        for y in range(self.input_shape[0]):
            for x in range(self.input_shape[1]):
                if np.array_equal(npy[y:y + self.new_size[0], x:x + self.new_size[1], :], sub_npy):
                    return y, x
        print('not found x, y')

    def testCrop(self):
        input_np = self.sess.run(self.input)
        for kind in ['left_top', 'left_bottom',
                     'mid', 'right_top', 'right_bottom']:
            out_ts = kits.crop(self.input, self.new_size, kind)
            out_np = self.sess.run(out_ts)
            y, x = self.indexof(input_np, out_np)
            print('pos: %s, coord: (%f, %f)' % (kind, y, x))

    def testCropAndResize(self):
        pass

    def testRandomFixCropWithMultiScale(self):
        ts = kits.random_fix_crop_with_multi_scale(
            self.input, [200, 100, 50], [224, 224])
        for i in range(10):
            npy = self.sess.run(ts)
            cv2.imwrite('%d.jpg' % i, npy)

    def test_subtract_mean(self):
        FLAGS['input'] = {}
        ts = tf.ones([2, 6])
        print('subtact_mean ts shape: ', ts.get_shape())
        print(self.sess.run(kits.subtract_mean(ts)))

if __name__ == '__main__':
    tf.test.main()
