import re

import tensorflow as tf
from config.config_agent import FLAGS, VARS


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    # following function will softmax internally
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss,
    #   plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    moving_average_decay = FLAGS['moving_average_decay']

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(moving_average_decay,
                                                      name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss;
    # do the same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of
        # the loss as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def _get_optimizer(lr):
    optimizer_list = [
        "momentum",
        "adam",
        # ""
    ]
    OPT = FLAGS['optimizer']
    name = OPT['name']
    args = OPT['args']

    if name not in optimizer_list:
        raise ValueError('%s optimizer not support', name)

    optimizer = getattr(tf.train, '%sOptimizer' % name.title())
    return optimizer(lr, **args)


def train(total_loss):
    """Train model graph.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
          processed.
    Returns:
        train_op: op for training.
    """
    global_step = VARS['global_step']
    decay_steps = FLAGS['num_steps_per_decay']
    initial_learning_rate = FLAGS['initial_learning_rate']
    learning_rate_decay_factor = FLAGS['decay_factor']
    moving_average_decay = FLAGS['moving_average_decay']

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = _get_optimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads,
                                            global_step=global_step)

    # # Add histograms for trainable variables.
    # for var in tf.trainable_variables():
    #     tf.histogram_summary(var.op.name, var)

    # # Add histograms for gradients.
    # for grad, var in grads:
    #     if grad is not None:
    #         tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def variable_on_cpu(name, shape, initializer, trainable=True):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, trainable=trainable,
                              initializer=initializer)
    return var


def variable_with_weight_decay(name, shape, stddev, wd, trainable=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """

    var = variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev),
        trainable=trainable)
    if wd is not None and VARS['mode'] == 'train':
        # wd_var = wd
        decay_factor = FLAGS['decay_factor']
        num_steps_per_decay = FLAGS['num_steps_per_decay']
        global_step = VARS['global_step']
        # Decay the learning rate exponentially based on the number of steps.
        wd_var = variable_on_cpu('weight_decay',
                                 [],
                                 initializer=tf.constant_initializer(wd),
                                 trainable=False)
        wd_var *= tf.pow(decay_factor,
                         tf.cast(global_step / num_steps_per_decay,
                                 tf.float32))
        # tf.scalar_summary('weight_decay', wd_var)
        # weight decay should decrease
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd_var, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # If a model is trained with multiple GPUs,
    # prefix all Op names with tower_name to differentiate the operations.
    # Note that this prefix is removed
    # from the names of the summaries when visualizing a model.

    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    TOWER_NAME = 'tower'
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
