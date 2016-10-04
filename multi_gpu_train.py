from __future__ import division

from datetime import datetime
import os
import sys
import traceback
import re
import time
from importlib import import_module

import numpy as np
from six.moves import xrange
import tensorflow as tf

# import local file
from input import input_agent
from config import config_agent
from config.config_agent import FLAGS, VARS
from model import kits

# module FLAGS
THIS = {}


def tower_loss(scope, nn, if_restart=False):

    moving_average_decay = FLAGS['moving_average_decay']

    inputs, labels, masks = input_agent.read()
    # Build inference Graph.
    logits = nn.inference(inputs, masks)
    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = nn.loss(logits, labels)
    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)
    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(moving_average_decay,
                                                      name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU
        # training session.
        # This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('tower_[0-9]*/', '', l.op.name)
        # Name each loss as '(raw)' and name the moving average version
        # of the loss as the original loss name.
        tf.scalar_summary(loss_name + ' (raw)', l)
        tf.scalar_summary(loss_name, loss_averages.average(l))

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


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
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def build_graph():
    # create a session and a coordinator
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    VARS['sess'] = tf.Session(config=config)
    VARS['coord'] = tf.train.Coordinator()
    VARS['global_step'] = tf.Variable(0, trainable=False)

    # real batch_size (temp)
    FLAGS['batch_size'] = int(FLAGS['batch_size'] / FLAGS['num_gpus'])


def launch_graph(nn, if_restart=False):

    sess = VARS['sess']
    global_step = VARS['global_step']

    batch_size = FLAGS['batch_size']
    train_dir = FLAGS['train_dir']
    decay_steps = FLAGS['num_steps_per_decay']
    initial_learning_rate = FLAGS['initial_learning_rate']
    learning_rate_decay_factor = FLAGS['decay_factor']
    moving_average_decay = FLAGS['moving_average_decay']
    num_gpus = FLAGS['num_gpus']
    max_steps = FLAGS['max_steps']

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = kits._get_optimizer(lr)

    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in xrange(num_gpus):
        with tf.device('/gpu:%d' % i), tf.name_scope('tower_%d' % i) as scope:
            # Calculate the loss for one tower of the model. This function
            # constructs the entire model but shares the variables across
            # all towers.
            loss = tower_loss(scope, nn, if_restart)


            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.scalar_summary('learning_rate', lr))

    # # Add histograms for gradients.
    # for grad, var in grads:
    #     if grad is not None:
    #         summaries.append(
    #             tf.histogram_summary(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # # Add histograms for trainable variables.
    # for var in tf.trainable_variables():
    #   summaries.append(tf.histogram_summary(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.merge_summary(summaries)

    # start read-threads
    input_agent.launch()

    # Build an initialization operation to run below.
    # init = tf.initialize_all_variables()
    # sess.run(init)
    # init alternatives
    var_lst = tf.all_variables()
    for var in var_lst:
        sess.run(tf.initialize_variables([var]))

    if if_restart is False:
        ckpt = tf.train.get_checkpoint_state(FLAGS['checkpoint_dir'])
        saver.restore(sess, ckpt.model_checkpoint_path)

    # Start the queue runners.

    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

    for step in xrange(sess.run(global_step), max_steps):
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss])
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
            num_examples_per_step = batch_size * num_gpus
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = duration / num_gpus
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), step, loss_value,
                                examples_per_sec, sec_per_batch))

        if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == max_steps:
            checkpoint_path = os.path.join(train_dir, 'model.ckpt')
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

        with tf.Graph().as_default(), tf.device('/cpu:0'):
            build_graph()
            try:
                launch_graph(model, if_restart)
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
