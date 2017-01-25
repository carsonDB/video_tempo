"""
train in a single GPU.
"""
from __future__ import division
import os
import time
from datetime import datetime
import numpy as np
from six.moves import xrange
import tensorflow as tf

from config import config_agent
from config.config_agent import FLAGS
from solver import Solver


class Train_solver(Solver):

    def __init__(self):
        self.lr_decay_iters = FLAGS['lr_decay_iters']
        self.initial_learning_rate = FLAGS['initial_learning_rate']
        self.max_steps = FLAGS['max_steps']
        self.step_per_summary = FLAGS['step_per_summary']
        self.step_per_ckpt = FLAGS['step_per_ckpt']
        super(Train_solver, self).__init__()

    def build_graph(self):
        # Build a Graph that computes the logits predictions.
        inputs = self.reader.read()
        # get optimizer
        opt = self.get_opt()

        with tf.device('/gpu:%d' % self.gpus[0]):
            # inference model.
            logits = self.model.infer(inputs['X'])
            # Calculate loss (cross_entropy and weights).
            self.loss = self.model.loss(logits, inputs['Y'])
            # Calculate the gradients for the batch of data
            grads = self.model.grad(opt, self.loss)
        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads,
                                                global_step=self.global_step)

        # Track the moving averages of all trainable variables.
        vars_averages = tf.train.ExponentialMovingAverage(
            self.moving_average_decay, self.global_step)
        vars_averages_op = vars_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, vars_averages_op]):
            self.train_op = tf.no_op(name='train')

    def launch_graph(self):
        start_step = self.sess.run(self.global_step)
        for step in xrange(start_step, self.max_steps):
            start_time = time.time()
            _, loss_value = self.sess.run([self.train_op, self.loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), "Model diverged with loss = NaN, "
            if step % 10 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ("%s: step %d, "
                              "loss = %.2f (%.1f examples/sec; %.3f "
                              "sec/batch) ")
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if step % self.step_per_summary == 0:
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % self.step_per_ckpt == 0 or (step + 1) == self.max_steps:
                checkpoint_path = os.path.join(self.dest_dir, 'model.ckpt')
                self.saver.save(self.sess, checkpoint_path, global_step=step)


def main(argv=None):
    # unroll arguments of train
    config_agent.init_FLAGS('train')
    Train_solver().start()

if __name__ == '__main__':
    tf.app.run()
