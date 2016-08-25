"""Build cnn model
Input:
    * batch of tensors
    * batch of labels
Return:
    * batch of logits
"""

import tensorflow as tf

# import local file
from config.config_agent import FLAGS


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
                              initializer=initializer, dtype=tf.float32)
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
        tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32),
        trainable)
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def affine_transform(inputs, dim1, scope_name, stddev=0.1, wd=0.0):
    # (w * x + b)
    dim0 = inputs.get_shape()[-1].value
    with tf.variable_scope(scope_name):
        weights = _variable_with_weight_decay('weights',
                                              shape=[dim0, dim1],
                                              stddev=stddev,
                                              wd=wd)
        biases = _variable_on_cpu('biases', [dim1],
                                  tf.constant_initializer(0.1))
    return tf.matmul(inputs, weights) + biases


def inference(inputs):
    """
    Inputs: [batch_size, num_step, *tf.get_shape(ly_out)]
    """
    INPUT = FLAGS['input']
    GRAPH = FLAGS['graph']
    batch_size = FLAGS['batch_size']
    num_step = INPUT['num_step']
    num_class = INPUT['num_class']
    hidden_size = GRAPH['hidden_size']
    num_layer = GRAPH['num_layer']

    # build LSTM subgraph
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0)
    # if config.keep_prob < 1:
    #   lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          # lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layer)
    # initial state
    initial_state = cell.zero_state(batch_size, tf.float32)
    # attend weights that dot product with output of pretrained model(e.g. cnn)
    in_channels = inputs.get_shape()[-1].value
    reshape_inputs = tf.reshape(inputs,
                                [batch_size, num_step, -1, in_channels])
    attend_weights_size = reshape_inputs.get_shape()[-2].value
    attend_weights_shape = [attend_weights_size, 1]
    initial_attend_weights = _variable_with_weight_decay('attend_weights',
                                                         attend_weights_shape,
                                                         stddev=0.01,
                                                         wd=0.0)

    # temporal build
    outputs = []
    state = initial_state
    attend_weights = initial_attend_weights
    for time_step in range(num_step):
        if time_step > 0:
            tf.get_variable_scope().reuse_variables()
        # attend weight inputs
        weighted_inputs = tf.reduce_sum(reshape_inputs[:, time_step, :]
                                        * attend_weights, 1)
        (cell_output, state) = cell(weighted_inputs, state)

        # produce next attend_weights
        attend_weights = affine_transform(cell_output, attend_weights_size,
                                          scope_name="attend")
        attend_weights = tf.nn.softmax(attend_weights)

        # output logits
        output = affine_transform(cell_output, num_class,
                                  scope_name="softmax_linear")
        outputs.append(tf.tanh(output))

    return tf.pack(outputs, axis=1)
