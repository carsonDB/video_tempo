"""feature_extract script
Any tensor storing in VARS['output'] will be available for output.
Running mode is the same with eval script (either valid or test).

All will be float32 by default

Output will be hdf5 files under ./data/.

"""
from __future__ import division
import os
import math
import numpy as np
import tensorflow as tf

from config import config_agent
from config.config_agent import FLAGS, VARS
from predict import Output_solver


class Test_solver(Output_solver):

    def __init__(self):
        super(Test_solver, self).__init__()
        self.output_path = './data/two_stream_test01.hdf5'

    def launch_graph(self):
        if self.num_examples % self.batch_size != 0:
            print("Warning: num_examples can't be divided by"
                  "batch_size with no remainder")
        # calculate iterations (all examples in test split)
        self.num_examples = self.reader.num_examples()
        num_iter = int(math.ceil(self.num_examples / self.batch_size))
        step = 0
        print('Total steps: %d' % num_iter)

        # write into a hdf5 file
        if tf.gfile.Exists(self.output_path):
            tf.gfile.Remove(self.output_path)
        tf.gfile.MakeDirs(os.path.dirname(self.output_path))

        # tensor to get
        scaled_logits = VARS['output']['scaled_logits']
        video_name = VARS['output']['name']
        video_label = VARS['output']['label']

        re_dict = {}
        video2class = {}
        while step < num_iter and not self.coord.should_stop():
            if step % 100 == 0:
                print('step: %d' % step)
            logits_npy, v_name, v_label = self.sess.run([scaled_logits,
                                                         video_name,
                                                         video_label])
            for i in range(self.batch_size):
                video2class[v_name[i]] = v_label[i]
                if v_name[i] in re_dict:
                    re_dict[v_name[i]] += logits_npy[i]
                else:
                    re_dict[v_name[i]] = logits_npy[i]

            step += 1

        # check results
        right_count = 0
        all_count = 0
        for v_n, v_l in re_dict.items():
            all_count += 1
            if np.argmax(v_l) == video2class[v_n]:
                right_count += 1
        print('test: %.4f' % (right_count / all_count))


def main(argv=None):
    # unroll arguments of prediction
    config_agent.init_FLAGS('test')
    Test_solver().start()

if __name__ == '__main__':
    tf.app.run()
