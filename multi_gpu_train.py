"""Multi_gpu_train
train on multi-gpus, in data-parallelism way
"""
from __future__ import division
from datetime import datetime
import os
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf

from config import config_agent
from config.config_agent import FLAGS
from solver import Solver


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable
        across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat_v2(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


class Multi_gpu_solver(Solver):

    def __init__(self):
        super(Multi_gpu_solver, self).__init__()
        self.max_steps = FLAGS['max_steps']
        self.decay_steps = FLAGS['num_steps_per_decay']
        self.initial_learning_rate = FLAGS['initial_learning_rate']
        self.gpus = FLAGS['gpus']
        self.num_gpus = len(self.gpus)
        assert self.batch_size % self.num_gpus == 0, (
            'Batch size must be divisible by number of GPUs')
        self.split_batch_size = self.batch_size // self.num_gpus

    def build_graph(self):
        # Build a Graph that computes the logits predictions.
        inputs = self.reader.read()
        # split into num_gpus groups
        inputs_lst = tf.split(inputs['X'], self.num_gpus, 0)
        labels_lst = tf.split(inputs['Y'], self.num_gpus, 0)
        # get optimizer
        opt = self.get_opt()
        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i in self.gpus:
                tower_name = 'tower_%d' % i
                with tf.device('/gpu:%d' % i), tf.name_scope(tower_name) as scope:
                    # inference model.
                    logits = self.model.infer(inputs_lst[i])
                    # Calculate loss (cross_entropy and weights).
                    self.loss = self.model.loss(logits, labels_lst[i], scope)
                    # add loss summary
                    tf.summary.scalar(self.loss.op.name, self.loss)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
                    # Retain the summaries from the final tower.
                    self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                       scope)
                    # Calculate the gradients for the batch of data
                    grads = opt.compute_gradients(self.loss)
                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        # Add a summary to track the learning rate.
        self.summaries.append(tf.summary.scalar('learning_rate',
                                                self.learning_rate))
        # Add a summary to track the queue state.
        self.summaries.append(tf.summary.scalar('queue',
                                                self.reader.queue_state))
        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads,
                                                global_step=self.global_step)

        # Track the moving averages of all trainable variables.
        vars_averages = tf.train.ExponentialMovingAverage(
            self.moving_average_decay, self.global_step)
        vars_averages_op = vars_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, vars_averages_op]):
            self.train_op = tf.no_op(name='train')

    def init_graph(self):
        # Build the summary operation based on the TF collection of Summaries.
        self.summary_op = tf.summary.merge(self.summaries)
        self.saver = tf.train.Saver(tf.global_variables())
        # initialize variables
        self.init_sess()
        # restore variables if any
        if self.if_restart is False:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        self.summary_writer = tf.summary.FileWriter(self.dest_dir,
                                                    self.sess.graph)

    def launch_graph(self):
        start_step = self.sess.run(self.global_step)
        for step in xrange(start_step, self.max_steps):
            start_time = time.time()
            _, loss_value = self.sess.run([self.train_op, self.loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % 10 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration

                format_str = ('%s: step %d, '
                              'loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == self.max_steps:
                checkpoint_path = os.path.join(self.dest_dir, 'model.ckpt')
                self.saver.save(self.sess, checkpoint_path, global_step=step)
                print('save to [%s] at global_step: %d' %
                      (checkpoint_path, step))


def main(argv=None):
    # unroll arguments of train
    config_agent.init_FLAGS('train')
    Multi_gpu_solver().start()

if __name__ == '__main__':
    tf.app.run()
