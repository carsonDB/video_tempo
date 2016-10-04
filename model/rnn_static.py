"""Build cnn model
Input:
    * batch of tensors
    * batch of labels
Return:
    * batch of logits
"""

import tensorflow as tf

# import local file
from config.config_agent import FLAGS, VARS
from model.kits import variable_on_cpu, variable_with_weight_decay
from model import kits


def affine_transform(inputs, dim1, scope_name, stddev=1, wd=0.0):
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


class Model():
    """rnn + attend
    """
    def __init__(self, flags=FLAGS):
        INPUT = flags['input']
        GRAPH = flags['graph']

        self.num_class = INPUT['num_class']
        self.hidden_size = GRAPH['hidden_size']
        self.num_layer = GRAPH['num_layer']
        self.keep_prob = GRAPH['dropout']
        self.num_step = INPUT['max_time_steps']
        self.batch_size = FLAGS['batch_size']

    def inference(self, inputs, masks):
        """
        Args:
            inputs: [batch_size, max_step, ...]
            masks: [batch_size, max_step]
        """

        self.masks = masks
        num_class = self.num_class
        num_layer = self.num_layer
        hidden_size = self.hidden_size
        keep_prob = self.keep_prob
        # raw_shape: [batch_size, max_step, ..., in_channels]
        raw_shape = inputs.get_shape().as_list()
        batch_size = raw_shape[0]
        num_step = raw_shape[1]
        channel_size = raw_shape[-1]

        inputs = tf.reshape(inputs, [batch_size, num_step, -1, channel_size])
        # inputs: [batch_size, max_step, feature, channel_size]
        feature_size = inputs.get_shape()[2].value

        # build LSTM subgraph
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)
        # dropout layer (at output)
        if keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=keep_prob)
        # multi-cells
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layer,
                                           state_is_tuple=True)

        output_lst = []
        state = cell.zero_state(batch_size, tf.float32)
        attend_weights = variable_with_weight_decay('first_attend_weights',
                                                    shape=[feature_size, 1],
                                                    stddev=1,
                                                    wd=None)
        cell_output = None
        with tf.variable_scope("RNN"):
            for time_step in range(num_step):
                if time_step > 0:
                    # get next attend_weights
                    attend_weights = affine_transform(cell_output,
                                                      feature_size,
                                                      'attend')
                    attend_weights = tf.nn.softmax(attend_weights)
                    attend_weights = tf.expand_dims(attend_weights, -1)

                    tf.get_variable_scope().reuse_variables()

                # next_input: [batch_size, feature, in_channels]
                next_input = inputs[:, time_step, :]
                next_input = tf.reduce_sum(next_input * attend_weights, 1)
                # next_input: [batch_size, in_channels]
                (cell_output, state) = cell(next_input, state)
                # cell_output: [batch_size, hidden_size]
                output_lst.append(cell_output)

        outputs = tf.pack(output_lst, 1)
        outputs = tf.tanh(outputs)
        outputs = affine_transform(outputs, num_class,
                                   scope_name="softmax_linear")
        # outputs: [batch_size, num_step, num_class]
        return outputs

    def loss(self, logits, labels):
        # labels: [batch_size, num_step]
        # logits: [batch_size, num_step, num_class]
        raw_shape = logits.get_shape().as_list()
        batch_size, num_step, num_class = raw_shape

        labels = tf.cast(labels, tf.int64)

        logits = tf.reshape(logits, [-1, num_class])
        labels = tf.reshape(labels, [-1])
        return kits.loss(logits, labels)

    def test(self, inputs, labels, masks, top):

        # Build a Graph that computes the logits predictions from the
        # inference model
        logits = self.inference(inputs, masks)
        # logits: [batch_size, num_step, num_class]
        avg_logits = tf.reduce_mean(logits, reduction_indices=1)
        # avg_logits: [batch_size, num_class]
        min_labels = tf.reduce_min(labels, reduction_indices=1)
        # max_labels = tf.reduce_max(labels, reduction_indices=1)
        # # for debug
        # VARS['equal_op'] = tf.equal(min_labels, max_labels)

        return tf.nn.in_top_k(avg_logits, min_labels, top)

