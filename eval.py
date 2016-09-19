"""Evaluation for model (single GPU).
"""

from importlib import import_module
from datetime import datetime
import math
import time
import traceback
import sys

import numpy as np
import tensorflow as tf

# import local file
from input import input_agent
from config import config_agent
from config.config_agent import FLAGS, VARS


THIS = {}


def eval_once(summary_writer, top_k_op):
    """Run Eval once.

    Args:
        summary_writer: Summary writer.
        top_k_op: Top K op.
        summary_op: Summary op.
    """
    batch_size = FLAGS['batch_size']
    num_examples = FLAGS['num_examples']
    num_top = FLAGS['top']

    sess = VARS['sess']
    coord = VARS['coord']
    saver = THIS['saver']

    ckpt = tf.train.get_checkpoint_state(FLAGS['checkpoint_dir'])
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        print('No checkpoint file found')
        return

    num_iter = int(math.ceil(1.0*num_examples / batch_size))
    true_count = 0  # Counts the number of correct predictions.
    total_sample_count = num_iter * batch_size
    step = 0
    print('Total steps: %d' % num_iter)
    while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        # print predictions
        true_count += np.sum(predictions)

        precision = 1.0*true_count / total_sample_count
        print('step: %d, precision %f' % (step, precision))

        step += 1

    # Compute precision @ 1.
    precision = 1.0*true_count / total_sample_count
    print('%s: precision @ %d = %.3f' % (datetime.now(), num_top, precision))

    summary = tf.Summary()
    # summary.ParseFromString(sess.run(summary_op))
    summary.value.add(tag='Precision @ %d' % num_top,
                      simple_value=precision)
    summary_writer.add_summary(summary, global_step)

    # # stop this time of evaluation
    # input_agent.pause()


def evaluate(nn):
    """Eval for one or a number of times."""

    # create a session and a coordinator
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    VARS['sess'] = sess = tf.Session(config=config)
    VARS['coord'] = tf.train.Coordinator()

    # Build subgraph (reader and preprocesser).
    inputs, labels = input_agent.read()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = nn.inference(inputs)

    # variables postfix with exponentialMovingAverage
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS['moving_average_decay'])
    variables_to_restore = variable_averages.variables_to_restore()
    THIS['saver'] = tf.train.Saver(variables_to_restore)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, FLAGS['top'])

    # Build the summary operation based on the TF collection of Summaries.
    summary_writer = tf.train.SummaryWriter(FLAGS['eval_dir'], sess.graph)

    # Start the queue runners.
    input_agent.launch()

    while True:
        eval_once(summary_writer, top_k_op)
        if FLAGS['run_once']:
            break
        time.sleep(FLAGS['eval_interval_secs'])


def main(argv=None):
    # unroll arguments of eval mode
    config_agent.init_FLAGS('eval')

    eval_dir = FLAGS['eval_dir']
    model_type = FLAGS['type']

    if tf.gfile.Exists(eval_dir):
        tf.gfile.DeleteRecursively(eval_dir)
    tf.gfile.MakeDirs(eval_dir)

    if model_type in ['cnn', 'rnn']:
        nn = import_module('model.' + model_type)
        with tf.Graph().as_default():

            try:
                evaluate(nn)
                print('eval process closed nomally\n')
            except:
                traceback.print_exc(file=sys.stdout)
                print('eval process closed with error\n')
            finally:
                input_agent.close()
    else:
        raise ValueError('no such model: %s', model_type)


if __name__ == '__main__':
    tf.app.run()
