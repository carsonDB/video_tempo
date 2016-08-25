"""Evaluation for model.
"""

from importlib import import_module
from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

# import local file
from input import input_agent
from config import config_agent
from config.config_agent import FLAGS


def eval_once(saver, summary_writer, top_k_op, summary_op,
              sess, coord,
              logits):
    """Run Eval once.

    Args:
        saver: Saver.
        summary_writer: Summary writer.
        top_k_op: Top K op.
        summary_op: Summary op.
    """
    batch_size = FLAGS['batch_size']
    num_examples = FLAGS['num_examples']

    with sess:
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

    try:
        # threads = []
        # for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        #     threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
        #                                      start=True))

        # Start the queue runners.
        # tf.train.start_queue_runners(sess=sess)

        num_iter = int(math.ceil(num_examples / batch_size))
        true_count = 0  # Counts the number of correct predictions.
        total_sample_count = num_iter * batch_size
        step = 0
        #debug
        # print sess.run([logits])
        while step < num_iter and not coord.should_stop():
            predictions = sess.run([top_k_op])
            # print predictions
            true_count += np.sum(predictions)
            step += 1

        # Compute precision @ 1.
        precision = true_count / total_sample_count
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op))
        summary.value.add(tag='Precision @ 1', simple_value=precision)
        summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

    # coord.request_stop()
    # coord.join(threads, stop_grace_period_secs=10)


def evaluate(nn):
    """Eval for one or a number of times."""

    with tf.Graph().as_default() as g:
        # Start process managers.
        sess = tf.Session()
        coord = tf.train.Coordinator()

        # Build subgraph (reader and preprocesser).
        inputs, labels, readers = input_agent.read(sess, coord)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = nn.inference(inputs)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS['moving_average_decay'])
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS['eval_dir'], g)

    while True:
        eval_once(saver, summary_writer, top_k_op, summary_op,
                  sess, coord, logits)
        if FLAGS['run_once']:
            break
        time.sleep(FLAGS['eval_interval_secs'])

    coord.request_stop()
    coord.join(readers)


def main(argv=None):
    # unroll arguments of eval mode
    config_agent.init_FLAGS('eval')
    eval_dir = FLAGS['eval_dir']
    model_type = FLAGS['type']

    if tf.gfile.Exists(eval_dir):
        tf.gfile.DeleteRecursively(eval_dir)
    tf.gfile.MakeDirs(eval_dir)

    if model_type in ['cnn', 'rnn']:
        evaluate(import_module('model.' + model_type))

if __name__ == '__main__':
    tf.app.run()
