"""
train in a single GPU.
"""

import os
import sys
import traceback
import time
from datetime import datetime
from importlib import import_module

import numpy as np
from six.moves import xrange
import tensorflow as tf

# import local file
from model import kits
from input import input_agent
from config import config_agent
from config.config_agent import FLAGS, VARS


# module FLAGS
THIS = {}


def build_graph(nn, if_restart=False):

    # create a session and a coordinator
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    VARS['sess'] = tf.Session(config=config)
    VARS['coord'] = tf.train.Coordinator()
    VARS['global_step'] = tf.Variable(0, trainable=False)

    # Build a Graph that computes the logits predictions
    inputs, labels, masks = input_agent.read()
    # inference model.
    logits = nn.inference(inputs, masks)
    # Calculate loss.
    loss = nn.loss(logits, labels)
    # updates the model parameters.
    train_op = kits.train(loss)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver(tf.all_variables())

    # THIS
    THIS['saver'] = saver
    THIS['loss'] = loss
    THIS['train_op'] = train_op
    THIS['summary_op'] = summary_op


def launch_graph(if_restart=False):

    sess = VARS['sess']
    global_step = VARS['global_step']

    loss = THIS['loss']
    train_op = THIS['train_op']
    summary_op = THIS['summary_op']
    saver = THIS['saver']

    # initialize variables and restore if any
    # init = tf.initialize_all_variables()
    # sess.run(init)
    var_lst = tf.all_variables()
    for var in var_lst:
        sess.run(tf.initialize_variables([var]))

    if if_restart is False:
        ckpt = tf.train.get_checkpoint_state(FLAGS['checkpoint_dir'])
        saver.restore(sess, ckpt.model_checkpoint_path)

    summary_writer = tf.train.SummaryWriter(FLAGS['train_dir'], sess.graph)
    # Start the queue runners.
    input_agent.launch()

    for step in xrange(sess.run(global_step), FLAGS['max_steps']):
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss])
        duration = time.time() - start_time

        assert not np.isnan(loss_value), "Model diverged with loss = NaN, "
        if step % 10 == 0:
            num_examples_per_step = FLAGS['batch_size']
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = ("%s: step %d, "
                          "loss = %.2f (%.1f examples/sec; %.3f "
                          "sec/batch) ")
            print(format_str % (datetime.now(), step, loss_value,
                                examples_per_sec, sec_per_batch))

        if step % 50 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 50 == 0 or (step + 1) == FLAGS['max_steps']:
            checkpoint_path = os.path.join(FLAGS['train_dir'],
                                           'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    # unroll arguments of train
    config_agent.init_FLAGS('train')

    train_dir = FLAGS['train_dir']
    model_type = FLAGS['type']
    if_restart = VARS['if_restart']

    if not tf.gfile.Exists(train_dir):
        # only start from scratch
        tf.gfile.MakeDirs(train_dir)
        if_restart = True

    elif if_restart:
        # clear old train_dir and restart
        tf.gfile.DeleteRecursively(train_dir)
        tf.gfile.MakeDirs(train_dir)

    if model_type in ['cnn', 'rnn', 'rnn_static']:
        nn_package = import_module('model.' + model_type)
        model = nn_package.Model()

        with tf.Graph().as_default():
            build_graph(model, if_restart)
            try:
                launch_graph(if_restart)
                print('training process closed normally\n')
            except:
                traceback.print_exc(file=sys.stdout)
                print('training process closed with error\n')
            finally:
                input_agent.close()

    else:
        raise ValueError('no such model: %s', model_type)

if __name__ == '__main__':
    tf.app.run()
