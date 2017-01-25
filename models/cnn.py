"""Build cnn model
Input:
    * batch of tensors
    * batch of labels
Return:
    * batch of logits
"""
from __future__ import division
import tensorflow as tf

from config.config_agent import FLAGS, VARS
from kits import variable_on_cpu, variable_with_weight_decay
from models.model_proto import Model_proto


def get_ly_name(ly, ly_id):
    return ly['name'] if 'name' in ly else (
        '%s_%d' % (ly['type'], ly_id))


def pad(ts, ly_padding):
    # padding in two ways
    if isinstance(ly_padding, (list, tuple)):
        ts = tf.pad(ts, [[0, 0]] + ly_padding + [[0, 0]])
        padding = 'VALID'
    elif isinstance(ly_padding, basestring):
        padding = ly_padding
    else:
        raise ValueError('padding wrong format', ly_padding)

    return ts, padding


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Perform the operation and get the output.
        op(self, *args, **kwargs)
        # Return self for chained calls.
        return self

    return layer_decorated


class Model(Model_proto):

    def __init__(self):
        super(Model, self).__init__()
        self.layers = FLAGS['layers']

    def infer(self, inputs):
        """Build the computation graph.

        * according to a list of layers' configuration.
        * self.layer_*(): can be alternative to any custom layers.
        * self.infer(): can be changed to composite layers customized.

        Args:
            inputs: batch of tensors
        Returns:
            unscaled Logits.
        """
        self.batch_size = inputs.get_shape().as_list()[0]
        layers = self.layers
        num_class = self.num_class
        self.ly_id = 0  # major id
        self.ly_out = inputs

        for i, ly in enumerate(layers):
            ly_type = ly['type']
            if not hasattr(self, 'layer_' + ly_type):
                raise ValueError('no such layer type: %s', ly_type)
            # compose chained layers
            getattr(self, 'layer_' + ly_type)(ly)

        # softmax, i.e. softmax(WX + b)
        with tf.variable_scope('softmax_linear') as scope:
            # last layer dimension == dim1 ? (last == fc)
            assert(len(self.ly_out.get_shape()) == 2)
            last_dim = self.ly_out.get_shape()[-1].value
            weights = variable_with_weight_decay('weights',
                                                 [last_dim, num_class],
                                                 stddev=(1 / last_dim),
                                                 wd=0.0)

            biases = variable_on_cpu('biases', [num_class],
                                     tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(self.ly_out, weights), biases,
                                    name=scope.name)
            # _activation_summary(softmax_linear)
        # don't softmax in advance
        return softmax_linear

    @layer
    def layer_conv2d(self, ly):
        self.ly_id += 1
        ly_out = self.ly_out
        with tf.variable_scope(get_ly_name(ly, self.ly_id)) as scope:
            k_shape = ly['filter'][:]
            # inner-channel (default -1)
            assert(k_shape[-2] == -1)
            k_shape[-2] = ly_out.get_shape()[-1].value

            kernel = variable_with_weight_decay('weights',
                                                shape=k_shape,
                                                stddev=ly[
                                                    'init_stddev'],
                                                wd=ly['weight_decay'])
            ly_out, ly_padding = pad(ly_out, ly['padding'])
            conv = tf.nn.conv2d(ly_out, kernel,
                                strides=ly['strides'],
                                padding=ly_padding)
            biases = variable_on_cpu('biases', [k_shape[-1]],
                                     tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            self.ly_out = tf.nn.relu(bias, name=scope.name)
            # _activation_summary(ly_out)

    @layer
    def layer_conv3d(self, ly):
        self.ly_id += 1
        ly_out = self.ly_out
        with tf.variable_scope(get_ly_name(ly, self.ly_id)) as scope:
            k_shape = ly['filter'][:]
            # inner-channel (default -1)
            assert(k_shape[-2] == -1)
            k_shape[-2] = ly_out.get_shape()[-1].value

            kernel = variable_with_weight_decay('weights',
                                                shape=k_shape,
                                                stddev=ly[
                                                    'init_stddev'],
                                                wd=ly['weight_decay'])
            ly_out, ly_padding = pad(ly_out, ly['padding'])
            conv = tf.nn.conv3d(ly_out, kernel,
                                strides=ly['strides'],
                                padding=ly_padding)
            biases = variable_on_cpu('biases', [k_shape[-1]],
                                     tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv, biases)
            self.ly_out = tf.nn.relu(bias, name=scope.name)
            # _activation_summary(ly_out)

    @layer
    def layer_max_pool2d(self, ly):
        ly_name = get_ly_name(ly, self.ly_id)
        ly_out = self.ly_out
        ly_out, ly_padding = pad(ly_out, ly['padding'])
        self.ly_out = tf.nn.max_pool(ly_out, ksize=ly['ksize'],
                                     strides=ly['strides'],
                                     padding=ly_padding,
                                     name=ly_name)

    @layer
    def layer_max_pool3d(self, ly):
        ly_name = get_ly_name(ly, self.ly_id)
        ly_out = self.ly_out
        ly_out, ly_padding = pad(ly_out, ly['padding'])
        self.ly_out = tf.nn.max_pool3d(ly_out, ksize=ly['ksize'],
                                       strides=ly['strides'],
                                       padding=ly_padding,
                                       name=ly_name)

    @layer
    def layer_lrn(self, ly):
        ly_name = get_ly_name(ly, self.ly_id)
        self.ly_out = tf.nn.lrn(self.ly_out, ly['depth_radius'],
                                alpha=ly['alpha'], beta=ly['beta'],
                                name=ly_name)

    @layer
    def layer_dropout(self, ly):
        if VARS['mode'] == 'train':
            self.ly_out = tf.nn.dropout(self.ly_out, keep_prob=ly['prob'])

    @layer
    def layer_fc(self, ly):
        self.ly_id += 1
        with tf.variable_scope(get_ly_name(ly, self.ly_id)) as scope:
            reshape = tf.reshape(self.ly_out, [self.batch_size, -1])
            dim0 = reshape.get_shape()[-1].value
            shape = ly['shape']
            dim1 = shape[-1]
            if shape[0] != -1 and shape[0] != dim0:
                raise ValueError('wrong dimension at fc-layer %d'
                                 % self.ly_id)

            weights = variable_with_weight_decay('weights',
                                                 shape=[dim0, dim1],
                                                 stddev=ly[
                                                     'init_stddev'],
                                                 wd=ly['weight_decay'])
            biases = variable_on_cpu('biases', [dim1],
                                     tf.constant_initializer(ly['init_bias']))
            self.ly_out = tf.nn.relu(tf.matmul(reshape, weights) + biases,
                                     name=scope.name)
            # _activation_summary(self.ly_out)
