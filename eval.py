"""Evaluation for model (single GPU).
"""
from __future__ import division
from datetime import datetime
import time
import math
import numpy as np
import tensorflow as tf

from config import config_agent
from config.config_agent import FLAGS, VARS
from solver import Solver


class Eval_solver(Solver):

    def __init__(self):
        self.num_examples = FLAGS['num_examples']
        self.num_top = FLAGS['top']
        self.run_once = FLAGS['run_once']
        self.eval_interval_secs = FLAGS['eval_interval_secs']
        self.if_test = VARS['if_test']
        super(Eval_solver, self).__init__()

    def build_graph(self):
        with tf.device('/gpu:%d' % self.gpus[-1]):
             # Build a Graph that computes the logits predictions.
            inputs = self.reader.read()
            # inference model.
            logits = self.model.infer(inputs['X'])
            # Calculate predictions.
            self.top_k_op = self.model.eval(logits, inputs['Y'], self.num_top)

    def init_graph(self):
        self.init_sess()
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            self.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        self.saver = tf.train.Saver(variables_to_restore)
        # Build the summary operation based on the TF collection of Summaries.
        self.summary_writer = tf.summary.FileWriter(self.dest_dir,
                                                    self.sess.graph)

    def launch_graph(self):
        while True:
            self._eval_once()
            if self.run_once:
                break
            time.sleep(self.eval_interval_secs)

    def _eval_once(self):
        """Eval once
        """
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = (ckpt.model_checkpoint_path
                           .split('/')[-1]
                           .split('-')[-1])
        else:
            print('No checkpoint file found')
            return

        num_iter = int(math.ceil(self.num_examples / self.batch_size))
        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * self.batch_size
        step = 0
        print('Total steps: %d' % num_iter)
        while step < num_iter and not self.coord.should_stop():
            predictions = self.sess.run([self.top_k_op])
            # print predictions
            true_count += np.sum(predictions)

            precision = true_count / total_sample_count
            # if step % 10 == 0:
            #     print('step: %d, precision %f' % (step, precision))

            step += 1

        # Compute precision @ 1.
        precision = true_count / total_sample_count
        print('%s: precision @ %d = %.3f'
              % (datetime.now(), self.num_top, precision))

        summary = tf.Summary()
        # summary.ParseFromString(sess.run(summary_op))
        summary.value.add(tag='Precision @ %d' % self.num_top,
                          simple_value=precision)
        self.summary_writer.add_summary(summary, global_step)


def main(argv=None):
    # unroll arguments of eval
    config_agent.init_FLAGS('eval')
    Eval_solver().start()

if __name__ == '__main__':
    tf.app.run()
