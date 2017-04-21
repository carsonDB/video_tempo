from __future__ import print_function
import re
import pprint
import tensorflow as tf

from config.config_agent import FLAGS, VARS


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


def variable_with_weight_decay(name, shape, stddev, wd=None,
                               weights_initializer=tf.truncated_normal_initializer,
                               trainable=True):
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

    if weights_initializer == tf.zeros_initializer:
        weights_initer = weights_initializer()
    else:
        weights_initer = weights_initializer(stddev=stddev)

    var = variable_on_cpu(
        name,
        shape,
        weights_initer,
        trainable=trainable)
    if wd is not None:
        weight_decay = tf.multiply(wd,
                                   tf.nn.l2_loss(var), name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def affine_transform(inputs, dim1, scope_name, stddev=0.1, wd=0.0):
    # (w * x + b)
    raw_shape = [-1 if i is None else i
                 for i in inputs.get_shape().as_list()]
    dim0 = raw_shape[-1]
    inputs = tf.reshape(inputs, [-1, dim0])

    with tf.variable_scope(scope_name):
        weights = variable_with_weight_decay('weights',
                                             shape=[dim0, dim1],
                                             stddev=stddev,
                                             wd=wd)
        biases = variable_on_cpu('biases', [dim1],
                                 tf.constant_initializer(0.1))
    output = tf.matmul(inputs, weights) + biases
    output = tf.reshape(output, raw_shape[:-1] + [dim1])

    return output


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
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


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
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def pp(*args, **kwargs):
    scope = kwargs.get('scope', None)
    last_scope = VARS.get('current_name_scope', None)
    # start a new section
    if scope is not None:
        current_scope = scope
        print('\n%s%s%s>' % ('-' * 6, current_scope, '-' * 6))
        VARS['current_name_scope'] = current_scope

    def stuff_to_str(stuff):
        s = ''
        for a in stuff:
            # tensor <name>: [shape]
            if isinstance(a, tf.Variable) or isinstance(a, tf.Tensor):
                s += '<%s>: %s; ' % (a.name, str(a.get_shape().as_list()))
            elif isinstance(a, (str, basestring)):
                s += a
            else:
                s += '%s; ' % str(a)
        return s

    print(stuff_to_str(args))


if __name__ == '__main__':
    tup = ('spam', ('eggs', ('lumberjack', ('knights',
                                            ('ni', ('dead', ('parrot', ('fresh fruit',))))))))
    stuff = ['a' * 10, tup, ['a' * 30, 'b' * 30], ['c' * 20, 'd' * 20]]
    print(stuff)
    pp(stuff)
