"""Build res101 model
"""
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf

from config.config_agent import FLAGS, VARS
from kits import variable_on_cpu, variable_with_weight_decay, pp
from models.model_proto import Model_proto
from models.slim_nets import resnet_v1


def w_loss(ts, weight_decay=0.0001):
    return tf.nn.l2_loss(ts) * weight_decay


def bn(ts, scope):
    is_training = (VARS['mode'] == 'train')
    with tf.variable_scope(scope):
        return tf.layers.batch_normalization(ts, momentum=0.997,
                                             training=is_training,
                                             epsilon=1e-5, name='BatchNorm')


def conv2d_bn_relu(ts, num_kernel, kernel_size, strides, activation=tf.nn.relu, name=None):
    ts = tf.layers.conv2d(ts, num_kernel, kernel_size,
                          strides=strides,
                          padding='same',
                          use_bias=False,
                          kernel_regularizer=w_loss,
                          name=name)
    ts = bn(ts, scope=name)

    pp('> layer %s: num_kernel %d, kernel_size %d, strides %d, activation: %s with bn\n'
       % (name, num_kernel, kernel_size, strides,
          activation.func_name if activation else 'none'), ts)
    return (activation(ts)
            if activation is not None
            else ts)


def pool2d(ts, kernel_size, strides, name):
    ts = tf.layers.max_pooling2d(
        ts, kernel_size, strides=strides, name=name)
    pp('> layer %s: kernel_size: %d, strides %d\n'
       % (name, kernel_size, strides), ts)

    return ts


def dense_drop(ts, units, dropout_prob, name):
    ts = tf.layers.dense(ts, units, activation=tf.nn.relu,
                         kernel_regularizer=w_loss, name=name)
    ts = tf.layers.dropout(ts, rate=dropout_prob,
                           training=(VARS['mode'] == 'train'))

    pp('> layer %s: units %d, dropout %.1f' % (name, units, dropout_prob), ts)
    return ts


def bottleneck_v1(ts, min_num_kernel, downsample=None, strides=1, activation=tf.nn.relu, name=None):
    """
    downsample == None: without shortcut_conv
    downsample == shortcut_conv
    """
    with tf.variable_scope('%s/bottleneck_v1' % name):
        raw_ts = (downsample(ts, 4 * min_num_kernel)
                  if downsample is not None else ts)

        ts = conv2d_bn_relu(ts, min_num_kernel, 1,
                            strides=strides, name='conv1')
        ts = conv2d_bn_relu(ts, min_num_kernel, 3, 1, name='conv2')
        ts = conv2d_bn_relu(ts, 4 * min_num_kernel, 1, 1,
                            activation=None, name='conv3')

        return activation(raw_ts + ts)


def block(ts, num_units, min_num_kernel, strides=2, name=None):

    def downsample(ts, num_kernel, kernel_size=1, strides=strides):
        # shortcut conv
        return conv2d_bn_relu(ts, num_kernel, kernel_size=kernel_size,
                              strides=strides, activation=None, name='shortcut')

    with tf.variable_scope(name):
        for unit_idx in range(num_units):
            unit_name = 'unit_%d' % (1 + unit_idx)
            if unit_idx == 0 and (strides != 1 or 64 != min_num_kernel * 4):
                ts = bottleneck_v1(ts, min_num_kernel,
                                   downsample=downsample, strides=strides, name=unit_name)
            else:
                ts = bottleneck_v1(ts, min_num_kernel,
                                   downsample=None, strides=1, name=unit_name)
    return ts


def resnet(ts, ly_lst, num_class, name):
    with tf.variable_scope(name):
        ts = conv2d_bn_relu(ts, 64, 7, 2, name='conv1')
        ts = pool2d(ts, 3, 2, name='pool1')
        ts = block(ts, ly_lst[0], 64, strides=1, name='block1')
        ts = block(ts, ly_lst[1], 128, name='block2')
        ts = block(ts, ly_lst[2], 256, name='block3')
        ts = block(ts, ly_lst[3], 512, name='block4')
        # global average pooling
        ts = tf.reduce_mean(ts, [1, 2], name='pool5')
        # linear fc
        ts = tf.layers.dense(ts, num_class, activation=None,
                             kernel_regularizer=w_loss,
                             name='logits')
    return ts


class Model(Model_proto):

    def __init__(self):
        super(Model, self).__init__()
        self.is_training = (VARS['mode'] == 'train')
        self.dropout_prob = self.model_config['dropout']

    def infer(self, inputs):
        input_shape = inputs.get_shape().as_list()
        self.input_batch_size = input_shape[0]
        pp('> res101 inputs', inputs)

        ts = resnet(inputs, ly_lst=[3, 4, 23, 3],
                    num_class=self.num_class, name='resnet_v1_101')
        pp('< res101 outputs', ts)

        return ts

    def get_vars_to_restore(self):

        def name_alter(name):
            return re.sub(r':\d+$', '', name).replace('kernel', 'weights') \
                .replace('bias', 'biases')

        vars_to_restore = {name_alter(var.name): var for var in tf.trainable_variables()
                           if var.name.startswith('resnet_v1_101') and 'logits' not in var.name}
        return vars_to_restore
