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


def train(nn):

    """Train for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # create a session and a coordinator
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=True))
        coord = tf.train.Coordinator()

        # Get clips and labels.
        inputs, labels, readers = input_agent.read(sess, coord)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = nn.inference(inputs)
        # Calculate loss.
        loss = kits.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = kits.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS['train_dir'], sess.graph)

        for step in xrange(FLAGS['max_steps']):
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
                              "sec/batch)")
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

        # finally stop readers
        coord.request_stop()
        coord.join(readers)


def main(argv=None):
    # unroll arguments of train
    config_agent.init_FLAGS('train')
    train_dir = FLAGS['train_dir']
    model_type = FLAGS['type']

    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)

    if model_type in ['cnn', 'rnn']:
        train(import_module('model.' + model_type))

if __name__ == '__main__':
    tf.app.run()
