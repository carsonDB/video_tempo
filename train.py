"""
train in a single GPU.
"""

from importlib import import_module
from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

# import local file
from model import kits
from input import input_agent
from config import config_agent
from config.config_agent import FLAGS


# module FLAGS
THIS = {}


def build_graph(nn, if_restart=False):

    # create a session and a coordinator
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    FLAGS['sess'] = sess = tf.Session(config=config)
    FLAGS['coord'] = coord = tf.train.Coordinator()

    if if_restart:
        global_step = tf.Variable(0, trainable=False)
    else:
        with sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS['checkpoint_dir'])
            THIS['ckpt'] = ckpt
        if not (ckpt and ckpt.model_checkpoint_path):
            raise ValueError('No checkpoint file found')
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1].split('-')[-1])
        global_step = tf.Variable(global_step, trainable=False)
    # update FLAGS
    FLAGS['global_step'] = global_step

    # Build a Graph that computes the logits predictions from the
    # Get clips and labels.
    inputs, labels = input_agent.read()

    # inference model.
    logits = nn.inference(inputs)

    # Calculate loss.
    loss = kits.loss(logits, labels)

    if not if_restart:
        # determine variables to restore
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS['moving_average_decay'])
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

    # updates the model parameters.
    train_op = kits.train(loss)
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # THIS
    THIS['coord'] = coord
    THIS['saver'] = saver
    THIS['loss'] = loss
    THIS['train_op'] = train_op
    THIS['summary_op'] = summary_op


def init_graph(if_restart=False):

    sess = FLAGS['sess']
    saver = THIS['saver']
    ckpt = THIS['ckpt']

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()
    # Start running operations on the Graph.
    sess.run(init)

    # train from scratch
    if if_restart:
        # Create a saver.
        THIS['saver'] = tf.train.Saver(tf.all_variables())
    else:
        saver.restore(sess, ckpt.model_checkpoint_path)


def launch_graph():

    sess = FLAGS['sess']
    train_op = THIS['train_op']
    loss = THIS['loss']
    summary_op = THIS['summary_op']
    saver = THIS['saver']
    global_step = FLAGS['global_step']

    summary_writer = tf.train.SummaryWriter(FLAGS['train_dir'], sess.graph)
    # Start the queue runners.
    # tf.train.start_queue_runners(sess=sess)

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

        if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == FLAGS['max_steps']:
            checkpoint_path = os.path.join(FLAGS['train_dir'],
                                           'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    # unroll arguments of train
    config_agent.init_FLAGS('train')
    train_dir = FLAGS['train_dir']
    model_type = FLAGS['type']
    if_restart = FLAGS['if_restart']

    if not tf.gfile.Exists(train_dir):
        # only start from scratch
        tf.gfile.MakeDirs(train_dir)
        if_restart = True

    elif tf.gfile.Exists(train_dir) and if_restart:
        # clear and restart
        tf.gfile.DeleteRecursively(train_dir)
        tf.gfile.MakeDirs(train_dir)

    if model_type in ['cnn', 'rnn']:

        with tf.Graph().as_default():
            nn = import_module('model.' + model_type)
            build_graph(nn, if_restart)
            init_graph(if_restart)
            try:
                launch_graph()
                print('training process closed normally')
            except:
                print('training process closed with error')
            finally:
                input_agent.close()

    else:
        raise ValueError('no such model: %s', model_type)

if __name__ == '__main__':
    tf.app.run()
