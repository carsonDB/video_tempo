"""Multi_gpu_train
train on multi-gpus, in data-parallelism way
"""
from __future__ import division
from __future__ import print_function
import os
import time
from datetime import datetime
import numpy as np
from six.moves import xrange
import tensorflow as tf

from config import config_agent
from config.config_agent import FLAGS, VARS
from kits import average_gradients, pp
from solver import Solver


class Multi_gpu_solver(Solver):

    def __init__(self):
        super(Multi_gpu_solver, self).__init__()
        self.max_steps = FLAGS['max_steps']
        self.step_per_summary = FLAGS['step_per_summary']
        self.step_per_ckpt = FLAGS['step_per_ckpt']
        self.gpus = FLAGS['gpus']
        self.num_gpus = len(self.gpus)
        assert self.input_batch_size % self.num_gpus == 0, (
            'Batch size must be divisible by number of GPUs')

    def build_graph(self):
        # Build a Graph that computes the logits predictions.
        inputs = self.reader.read()
        # split into num_gpus groups
        inputs_lst = tf.split(inputs['X'], self.num_gpus, 0)
        labels_lst = tf.split(inputs['Y'], self.num_gpus, 0)
        # get optimizer
        opt = self.model.get_opt()

        # Calculate the gradients for each model tower.
        tower_losses = []
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i, gpu_idx in enumerate(self.gpus):
                tower_name = 'tower_%d' % i
                with tf.device('/gpu:%d' % gpu_idx), tf.name_scope(tower_name) as scope:
                    pp(scope=tower_name)
                    # inference model.
                    logits = self.model.infer(inputs_lst[i])
                    # Calculate loss (cross_entropy and weights).
                    loss = self.model.loss(logits, labels_lst[i], scope)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()
                    if i == 0:
                        # add loss summary (one gpu of one of iter_size)
                        tf.summary.scalar(loss.op.name, loss)
                        # Retain the summaries from the final tower.
                        self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES,
                                                           scope)
                    # Calculate the gradients for the batch of data
                    grads = opt.compute_gradients(loss)
                    # Keep track of the loss and grads across all towers.
                    tower_losses.append(loss)
                    tower_grads.append(grads)

        # Calculate mean of losses and grads across all towers.
        loss = tf.reduce_mean(tower_losses)
        grad_var_lst = average_gradients(tower_grads)

        # iter_size batch
        grads = [grad_var[0] for grad_var in grad_var_lst]
        loss_grads = tf.train.batch([loss] + grads,
                                    batch_size=self.iter_size)
        self.loss = tf.reduce_mean(loss_grads[0], axis=0)
        ave_grads = [(tf.reduce_mean(grad, axis=0), grad_var_lst[i][1])
                     for i, grad in enumerate(loss_grads[1:])]

        # update moving_average (e.g. moving_mean of batch_norm)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(ave_grads,
                                                global_step=self.global_step)

        # Add a summary to track the learning rate.
        self.summaries.append(tf.summary.scalar('learning_rate',
                                                self.model.learning_rate))

        with tf.control_dependencies(update_ops + [apply_gradient_op]):
            self.train_op = tf.no_op(name='train')

    def launch_graph(self):
        start_step = self.run_sess(self.global_step)
        for step in xrange(start_step, self.max_steps):
            start_time = time.time()
            _, loss_value = self.run_sess([self.train_op, self.loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), "Model diverged with loss = NaN, "

            if self.run_mode == 'debug' or step % 10 == 0:
                queue_state = self.run_sess(self.reader.queue_state) * 100
                lr = self.run_sess(self.model.learning_rate)
                if queue_state < 10 or self.run_mode == 'debug':
                    print('queue state: %.0f%%, learning_rate: %f' %
                          (queue_state, lr))

            if step % 10 == 0:
                examples_per_sec = self.batch_size / duration
                sec_per_batch = float(duration)

                format_str = ("%s: step %d, "
                              "loss = %.4f (%.1f examples/sec; %.3f "
                              "sec/batch) ") % (datetime.now(), step, loss_value,
                                                examples_per_sec, sec_per_batch)
                print(format_str)

            if step % self.step_per_summary == 0:
                summary_str = self.run_sess(self.summary_op)
                self.summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % self.step_per_ckpt == 0 or (step + 1) == self.max_steps:
                checkpoint_path = os.path.join(self.dest_dir, 'model.ckpt')
                self.saver.save(self.sess, checkpoint_path, global_step=step)
                print('> save to "%s" at global_step: %d\n' %
                      (checkpoint_path, step))


def main(argv=None):
    # unroll arguments of train
    config_agent.init_FLAGS('train')
    VARS['mode'] = 'train'
    Multi_gpu_solver().start()

if __name__ == '__main__':
    tf.app.run()
