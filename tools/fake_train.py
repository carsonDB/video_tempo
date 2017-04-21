"""Multi_gpu_train
train with input from caffe
"""
from __future__ import division
import os
import time
from datetime import datetime
import numpy as np
import csv
import h5py
from six.moves import xrange
import tensorflow as tf

from config import config_agent
from config.config_agent import FLAGS, VARS
from kits import average_gradients
from solver import Solver


data_lst = '/home/wsy/dataset/tmp/caffe_train.txt'
DATA = None


def load_data_lst(data_lst):
    lst = []
    with open(data_lst) as f:
        for row in csv.reader(f, delimiter=' '):
            lst.append(row[0])
    global DATA
    DATA = lst


def fetch_data(batch_size):
    num_hdf5 = batch_size // 16
    out_npy = []
    out_label = []
    for path in DATA[:num_hdf5]:
        with h5py.File(path) as f:
            out_npy.append(np.transpose(
                np.array(f['data']).astype('float32'), [0, 2, 3, 1]))
            out_label.append(np.array(f['label']).astype('int32').reshape([-1]))
    # print('out_npy', out_npy.shape)
    out_npy = np.concatenate(out_npy, axis=0)
    out_label = np.concatenate(out_label, axis=0)
    DATA[:num_hdf5] = []

    return {'X': out_npy, 'Y': out_label}


class Multi_gpu_solver(Solver):

    def __init__(self):
        super(Multi_gpu_solver, self).__init__()
        self.max_steps = FLAGS['max_steps']
        self.decay_steps = FLAGS['num_steps_per_decay']
        self.initial_learning_rate = FLAGS['initial_learning_rate']
        self.step_per_summary = FLAGS['step_per_summary']
        self.step_per_ckpt = FLAGS['step_per_ckpt']
        self.gpus = FLAGS['gpus']
        self.num_gpus = len(self.gpus)
        assert self.input_batch_size % self.num_gpus == 0, (
            'Batch size must be divisible by number of GPUs')

    def build_graph(self):
        # Build a Graph that computes the logits predictions.
        _inputs = self.reader.read()
        inputs = {
            'X': tf.placeholder(tf.float32, shape=_inputs['X'].get_shape()),
            'Y': tf.placeholder(tf.int32, shape=_inputs['Y'].get_shape())
        }
        self.inputs = inputs
        # split into num_gpus groups
        inputs_lst = tf.split(inputs['X'], self.num_gpus, 0)
        labels_lst = tf.split(inputs['Y'], self.num_gpus, 0)
        # get optimizer
        opt = self.get_opt()

        # Calculate the gradients for each model tower.
        tower_losses = []
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i in self.gpus:
                tower_name = 'tower_%d' % i
                with tf.device('/gpu:%d' % i), tf.name_scope(tower_name) as scope:
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
                    grad_var_lst = opt.compute_gradients(loss)
                    # Keep track of the loss and grads across all towers.
                    tower_losses.append(loss)
                    tower_grads.append(grad_var_lst)

        # calculate mean of losses and grads across all towers.
        self.loss = tf.reduce_mean(tower_losses)
        grad_var_lst = average_gradients(tower_grads)

        # Apply gradients.
        self.placeholder_grads = []
        self.grads = []
        for grad, var in grad_var_lst:
            self.grads.append(grad)
            self.placeholder_grads.append(
                (tf.placeholder(tf.float32, shape=var.get_shape()), var))
        apply_gradient_op = opt.apply_gradients(
            self.placeholder_grads, global_step=self.global_step)

        # Add a summary to track the learning rate.
        self.summaries.append(tf.summary.scalar('learning_rate',
                                                self.learning_rate))

        with tf.control_dependencies([apply_gradient_op]):
            self.train_op = tf.no_op(name='train')

    def launch_graph(self):
        start_step = self.sess.run(self.global_step)
        for step in xrange(start_step, self.max_steps):
            start_time = time.time()
            # every iter_size inputs and that counts as 1 iteration.
            loss_lst, grads_lst = [], []
            for sub_iter in range(self.iter_size):

                feed_dict = {}
                data_dict = fetch_data(
                    self.inputs['X'].get_shape().as_list()[0])
                feed_dict[self.inputs['X']] = data_dict['X']
                feed_dict[self.inputs['Y']] = data_dict['Y']

                loss_value, grads_value = self.sess.run(
                    [self.loss, self.grads], feed_dict=feed_dict)
                loss_lst.append(loss_value)
                grads_lst.append(grads_value)
            # average losses and grads
            avg_loss_value = np.mean(loss_lst)
            avg_grads_value = np.mean(grads_lst, axis=0)
            feed_dict = {
                grad_var[0]: avg_grads_value[i]
                for i, grad_var in enumerate(self.placeholder_grads)}

            _ = self.sess.run([self.train_op], feed_dict=feed_dict)
            duration = time.time() - start_time

            assert not np.isnan(
                avg_loss_value), "Model diverged with loss = NaN, "

            if self.run_mode == 'debug' and step % 10 == 0:
                queue_state = self.sess.run(self.reader.queue_state * 100)
                lr = self.sess.run(self.learning_rate)
                print('queue state: %.0f%%, learning_rate: %f' %
                      (queue_state, lr))

            if step % 10 == 0:
                examples_per_sec = self.batch_size / duration
                sec_per_batch = float(duration)

                format_str = ("%s: step %d, "
                              "loss = %.4f (%.1f examples/sec; %.3f "
                              "sec/batch) ")
                print(format_str % (datetime.now(), step, avg_loss_value,
                                    examples_per_sec, sec_per_batch))

            # Save the model checkpoint periodically.
            if step % self.step_per_ckpt == 0 or (step + 1) == self.max_steps:
                checkpoint_path = os.path.join(self.dest_dir, 'model.ckpt')
                self.saver.save(self.sess, checkpoint_path,
                                global_step=step)
                print('> save to "%s" at global_step: %d\n' %
                      (checkpoint_path, step))


def main(argv=None):
    # unroll arguments of train
    config_agent.init_FLAGS('train')
    VARS['mode'] = 'train'
    load_data_lst(data_lst)
    FLAGS['input_queue']['capacity'] = 1
    Multi_gpu_solver().start()

if __name__ == '__main__':
    tf.app.run()
