"""extract ucf101_rgb data for training
"""
from __future__ import division
import sys
import numpy as np
import traceback
import h5py
import tensorflow as tf

from config import config_agent
from config.config_agent import FLAGS, VARS
from solver import Solver


class Fake_reader(Solver):

    def __init__(self):
        super(Fake_reader, self).__init__()
        self.hdf5_path = '/home/wsy/dataset/tmp/tempo_train_$.hdf5'
        self.txt_path = '/home/wsy/dataset/tmp/tempo_train.txt'
        self.all_count = 256 * 1200

    def start(self):
        with tf.Graph().as_default() as self.graph, tf.device('/cpu:0'):
            try:
                self.init_env()
                self.inputs = self.reader.read()
                self.reader.launch()
                self.read_into_hdf5()
                print('%s process closed normally\n' % VARS['mode'])
            except:
                traceback.print_exc(file=sys.stdout)
                print('%s process closed with error\n' % VARS['mode'])
            finally:
                self.reader.close()

    def read_into_hdf5(self):
        max_steps = self.all_count // self.input_batch_size
        print('total steps: %d' % max_steps)
        records = ''

        for step in xrange(max_steps):
            hdf5_path = self.hdf5_path.replace('$', '%d' % step)
            f = h5py.File(hdf5_path, 'w')
            f.create_dataset('data', (self.input_batch_size, 3, 224, 224))
            f.create_dataset('label', (self.input_batch_size, 1, 1, 1))
            inputs = self.sess.run(self.inputs)
            assert(inputs['X'].shape[0] == self.input_batch_size)
            f['data'][:] = np.transpose(inputs['X'], [0, 3, 1, 2])[:]
            f['label'][:] = inputs['Y'].reshape(
                [self.input_batch_size, 1, 1, 1])[:]
            f.close()
            records += '%s\n' % hdf5_path
            if step % 1 == 0:
                print(step)

        with open(self.txt_path, 'w') as f:
            f.write(records)

if __name__ == '__main__':
    config_agent.init_FLAGS('train')
    VARS['mode'] = 'train'
    Fake_reader().start()
