"""feature_extract script

All will be float32 by default

"""
from __future__ import division
from __future__ import print_function
import math
import progressbar
import numpy as np
import tensorflow as tf

from config import config_agent
from config.config_agent import FLAGS, VARS
from kits import pp
from solver import Solver


class Test_solver(Solver):

    def __init__(self):
        super(Test_solver, self).__init__()
        self.num_examples = FLAGS['num_examples']
        self.num_top = FLAGS['top']
        self.run_once = FLAGS['run_once']
        self.gpus = FLAGS['gpus']
        self.eval_interval_secs = FLAGS['eval_interval_secs']
        self.if_test = VARS['if_test']
        self.out_lst = {}

    def build_graph(self):
        # Build a Graph that computes the logits predictions.
        inputs = self.reader.read()
        # Build a Graph that computes the logits predictions.
        with tf.device('/gpu:%d' % self.gpus[0]):
            # inference model.
            logits = self.model.infer(inputs['X'])
            self.out_lst['name'] = inputs['name']
            self.out_lst['label'] = inputs['Y']
            # scale logits
            self.out_lst['logits'] = logits
            self.out_lst['scaled_logits'] = tf.nn.softmax(logits)

    def init_graph(self):
        # self.init_sess()
        saver = tf.train.Saver(tf.global_variables())
#        # temp: test caffemode-10000
#        print('temp init: %s' % self.model.init_weights_path)
#        self.model.model_init_from_hdf5()

        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('restored from :', ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return

    def launch_graph(self):
        # calculate iterations (all examples in test split)
        self.num_examples = self.reader.num_examples()
        # self.num_examples = 100000
        print('num of examples: %d' % self.num_examples)

        if self.num_examples % self.input_batch_size != 0:
            print("Warning: num_examples can't be divided by"
                  "input_batch_size with no remainder")

        num_iter = int(math.ceil(self.num_examples / self.input_batch_size))
        print('actually covered num_examples: %d' %
              (num_iter * self.input_batch_size))
        print('Total steps: %d' % num_iter)

        # tensor to get
        # logits = self.out_lst['logits']
        pp('logits -> softmax -> sum')
        logits = self.out_lst['scaled_logits']
        video_name = self.out_lst['name']
        video_label = self.out_lst['label']

        re_dict = {}
        video2class = {}
        re_count = {}

        bar = progressbar.ProgressBar()
        for step in bar(range(num_iter)):
            if self.coord.should_stop():
                break
            logits_npy, v_name, v_label = self.sess.run([logits,
                                                         video_name,
                                                         video_label])
            for i in range(self.input_batch_size):
                video2class[v_name[i]] = v_label[i]
                if v_name[i] in re_dict:
                    re_dict[v_name[i]] += logits_npy[i]
                    re_count[v_name[i]] += 1
                else:
                    re_dict[v_name[i]] = logits_npy[i]
                    re_count[v_name[i]] = 1

        # def softmax(x):
        #     """Compute softmax values for each sets of scores in x."""
        #     return np.exp(x) / np.sum(np.exp(x), axis=0)

        # check results
        right_count = 0
        all_count = 0
        for v_n, v_l in re_dict.iteritems():
            all_count += 1
            if re_count[v_n] != self.reader.num_per_video * 10:
                pp('waring re_count[%s] == %d' % (v_n, re_count[v_n]))
            scaled_logits = v_l / re_count[v_n]
            if np.argmax(scaled_logits) == video2class[v_n]:
                right_count += 1

        print('test accuracy: %.4f\nall counts: %d\nall_examples: %d' %
              (right_count / all_count, all_count, self.num_examples))


def main(argv=None):
    # unroll arguments of prediction
    config_agent.init_FLAGS('eval')
    VARS['mode'] = 'test'
    Test_solver().start()

if __name__ == '__main__':
    tf.app.run()
