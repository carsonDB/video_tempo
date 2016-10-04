"""Build cnn model
Input:
    * batch of tensors
    * batch of labels
Return:
    * batch of logits
"""

import tensorflow as tf
from tensorflow.python.ops.rnn import raw_rnn

# import local file
from config.config_agent import FLAGS, VARS
from model.kits import variable_on_cpu, variable_with_weight_decay


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

    def inference(self, inputs, masks):
        """
        Args:
            inputs: [batch_size, max_step, ...]
            masks: [batch_size, max_step]
        """

        self.masks = masks
        num_class = self.num_class
        hidden_size = self.hidden_size
        num_layer = self.num_layer
        keep_prob = self.keep_prob
        seq_length = tf.reduce_sum(tf.cast(masks, tf.int32), -1)

        # raw_shape: [batch_size, max_step, ..., in_channels]
        raw_shape = inputs.get_shape().as_list()
        batch_size = raw_shape[0]
        num_step = raw_shape[1]
        in_channel = raw_shape[-1]

        inputs = tf.reshape(inputs, [batch_size, num_step, -1, in_channel])
        # inputs : [batch_size, max_step, feature, in_channels]
        feature_size = inputs.get_shape()[2].value
        channel_size = inputs.get_shape()[3].value

        def loop_fn(time, cell_output, loop_state):
            if cell_output is None:
                # time == 0
                emit_output = None
                # attend weights that dot product with inputs
                # init first attention
                attend_shape = [batch_size, feature_size, 1]
                attend_weights = variable_with_weight_decay('attend_weights',
                                                            attend_shape,
                                                            stddev=0.01,
                                                            wd=0.0)
            else:
                emit_output = cell_output
                # attention from last output
                attend_weights = affine_transform(cell_output,
                                                  feature_size,
                                                  scope_name='attend')
                attend_weights = tf.nn.softmax(attend_weights)
                attend_weights = tf.expand_dims(attend_weights, -1)

            elements_finished = (time >= seq_length)
            finished = tf.reduce_all(elements_finished)
            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, feature_size],
                                 dtype=tf.float32),
                lambda: tf.reduce_sum(inputs_ta.read(time) * attend_weights, 2)
                # lambda: tf.reduce_mean(inputs_ta.read(time), 2)
            )

            next_loop_state = None
            return (elements_finished, next_input,
                    emit_output, next_loop_state)

        # build LSTM subgraph
        lstm_cell = tf.nn.rnn_cell.LSTMCell(hidden_size,
                                            state_is_tuple=True)
        # dropout layer (at output)
        if keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=keep_prob)
        # multi-cells
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layer,
                                           state_is_tuple=True)

        # dynamic_rnn
        # inputs: [batch_size, max_step, ...]
        inputs = tf.transpose(inputs, perm=[1, 0, 2, 3])
        # inputs: [max_step, batch_size, ...]
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=num_step)
        inputs_ta = inputs_ta.unpack(inputs)

        # init (? ...)
        initial_state = cell.zero_state(batch_size, tf.float32)
        # start to run overall
        outputs_ta, final_state, _ = raw_rnn(cell, loop_fn, initial_state)
        outputs = outputs_ta.pack()
        # outputs: [max_step, batch_size, hidden_size]
        # softmax_linear (? ...)
        outputs = tf.tanh(outputs)
        outputs = affine_transform(outputs, num_class,
                                   scope_name="softmax_linear")
        # outputs: [max_step, batch_size, num_class]
        outputs = tf.transpose(outputs, perm=[1, 0, 2])
        # outputs: [batch_size, max_step, num_class]

        return outputs

    def loss(self, logits, labels):
        masks = self.masks

        # logits: [batch_size, max_step, num_class]
        raw_shape = logits.get_shape().as_list()
        # labels: [batch_size]
        # masks: [batch_size, max_step]

        batch_size, max_step, num_class = raw_shape
        max_step = self.num_step
        seq_length = tf.reduce_sum(masks, 1)

        # labels: [batch_size, max_step]
        labels = tf.cast(labels, tf.int64)
        # following function will softmax internally
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.reshape(logits, [-1, num_class]),
            tf.reshape(labels, [-1]),
            name='cross_entropy_per_example')
        cross_entropy = tf.reshape(cross_entropy * tf.reshape(masks, [-1]),
                                   [batch_size, max_step])
        # cross_entropy: [batch_size, max_step]
        cross_entropy_sum = tf.reduce_sum(cross_entropy, 1,
                                          name='cross_entropy_sum')
        cross_entropy_mean = tf.reduce_mean(cross_entropy_sum / seq_length)
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss,
        #   plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')
