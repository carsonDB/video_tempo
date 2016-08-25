"""Build cnn model
Input:
    * batch of tensors
    * batch of labels
Return:
    * batch of logits
"""

import re
import tensorflow as tf

# import local file
from config.config_agent import FLAGS


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer, trainable=True):
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


def _variable_with_weight_decay(name, shape, stddev, wd, trainable=True):
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
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev),
        trainable=trainable)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def inference(inputs, flags=None, output_id=None):
    """Build the computation graph.

    Args:
      inputs: batch of tensors
      flag: a dict set for specific situation
      output_id: return specific layer

    Returns:
      unscaled Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables
    #   across multiple GPU training runs.
    # If we only ran this model on a single GPU,
    #   we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #

    # set specific FLAGS namespace (pretrained model)
    if flags is not None:
        trainable = False
    else:
        flags = FLAGS
        trainable = True

    num_class = flags['input']['num_class']
    batch_size = flags['batch_size']
    layers = flags['graph']
    ly_id = 0
    ly_out = inputs

    for i, ly in enumerate(layers):
        ly_type = ly['type']

        # conv_2d layer
        if ly_type == 'conv2d':
            ly_id += 1
            with tf.variable_scope('conv2d_%d' % ly_id) as scope:
                k_shape = ly['filter']
                k_shape[-2] = ly_out.get_shape()[-1].value
                kernel = _variable_with_weight_decay('weights',
                                                     shape=k_shape,
                                                     stddev=ly['init_stddev'],
                                                     wd=ly['weight_decay'],
                                                     trainable=trainable)
                conv = tf.nn.conv2d(ly_out, kernel,
                                    strides=ly['strides'],
                                    padding=ly['padding'])
                biases = _variable_on_cpu('biases', [k_shape[-1]],
                                          tf.constant_initializer(0.0),
                                          trainable=trainable)
                bias = tf.nn.bias_add(conv, biases)
                ly_out = tf.nn.relu(bias, name=scope.name)
                _activation_summary(ly_out)

        # conv_3d layer
        elif ly_type == 'conv3d':
            ly_id += 1
            with tf.variable_scope('conv3d_%d' % ly_id) as scope:
                k_shape = ly['filter']
                k_shape[-2] = ly_out.get_shape()[-1].value
                kernel = _variable_with_weight_decay('weights',
                                                     shape=k_shape,
                                                     stddev=ly['init_stddev'],
                                                     wd=ly['weight_decay'],
                                                     trainable=trainable)
                conv = tf.nn.conv3d(ly_out, kernel,
                                    strides=ly['strides'],
                                    padding=ly['padding'])
                biases = _variable_on_cpu('biases', [k_shape[-1]],
                                          tf.constant_initializer(0.0),
                                          trainable=trainable)
                bias = tf.nn.bias_add(conv, biases)
                ly_out = tf.nn.relu(bias, name=scope.name)
                _activation_summary(ly_out)

        # max_pool_2d layer
        elif ly_type == 'max_pool2d':
            ly_out = tf.nn.max_pool2d(ly_out, ksize=ly['ksize'],
                                      strides=ly['strides'],
                                      padding=ly['padding'],
                                      name=('pool2d_%d' % ly_id))

        # max_pool_3d layer
        elif ly_type == 'max_pool3d':
            ly_out = tf.nn.max_pool3d(ly_out, ksize=ly['ksize'],
                                      strides=ly['strides'],
                                      padding=ly['padding'],
                                      name=('pool3d_%d' % ly_id))

        elif ly_type == 'dropout':
            ly_out = tf.nn.dropout(ly_out, keep_prob=ly['prob'])

        # full connected layer
        elif ly_type == 'fc':
            ly_id += 1
            with tf.variable_scope('local_%d' % ly_id) as scope:
                reshape = tf.reshape(ly_out, [batch_size, -1])
                dim0 = reshape.get_shape()[-1].value
                shape = ly['shape']
                dim1 = shape[-1]
                if shape[0] != -1 and shape[0] != dim0:
                    raise ValueError('wrong dimension at fc-layer %d' % ly_id)

                weights = _variable_with_weight_decay('weights',
                                                      shape=[dim0, dim1],
                                                      stddev=ly['init_stddev'],
                                                      wd=ly['weight_decay'],
                                                      trainable=trainable)
                biases = _variable_on_cpu('biases', [dim1],
                                          tf.constant_initializer(0.1),
                                          trainable=trainable)
                ly_out = tf.nn.relu(tf.matmul(reshape, weights) + biases,
                                    name=scope.name)
                _activation_summary(ly_out)
        else:
            raise ValueError('no such layer: %s' % ly_type)

        # if output layer is specific
        if i == output_id:
            return ly_out

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        # last layer dimension == dim1 ? (last == fc)
        last_dim = ly_out.get_shape()[-1].value
        weights = _variable_with_weight_decay('weights',
                                              [last_dim, num_class],
                                              stddev=0.01,
                                              wd=0.0,
                                              trainable=trainable)

        biases = _variable_on_cpu('biases', [num_class],
                                  tf.constant_initializer(0.1),
                                  trainable=trainable)
        softmax_linear = tf.add(tf.matmul(ly_out, weights), biases,
                                name=scope.name)
        _activation_summary(softmax_linear)
    # don't softmax in advance

    return softmax_linear
