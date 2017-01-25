import re
import tensorflow as tf

from config.config_agent import FLAGS


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
