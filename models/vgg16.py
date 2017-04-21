"""Build vgg model
"""
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf

from config.config_agent import FLAGS, VARS
from kits import variable_on_cpu, variable_with_weight_decay, pp
from models.model_proto import Model_proto


class Model(Model_proto):

    def __init__(self):
        super(Model, self).__init__()
        self.is_training = (VARS['mode'] == 'train')
        self.dropout_prob = self.model_config['dropout']

    def infer(self, inputs):
        input_shape = inputs.get_shape().as_list()
        self.input_batch_size = input_shape[0]

        def w_loss(ts, weight_decay=0.0005):
            return tf.nn.l2_loss(ts) * weight_decay

        def reshape_rank2(ts):
            # shape [-1, last dimension]
            ts_shape = ts.get_shape().as_list()
            return tf.reshape(ts, [ts_shape[0], -1])

        def conv2d(ts, num_kernel, kernel_size=3, padding='same', name=None):
            return tf.layers.conv2d(ts, num_kernel, kernel_size,
                                    activation=tf.nn.relu,
                                    padding=padding,
                                    kernel_regularizer=w_loss,
                                    name=name)

        def pool2d(ts, name):
            return tf.layers.max_pooling2d(ts, [2, 2], strides=2, name=name)

        def dropout(ts):
            return tf.layers.dropout(
                ts, rate=self.dropout_prob, training=self.is_training)

        def dense(ts, units, name):
            ts = tf.layers.dense(ts, units, activation=tf.nn.relu,
                                 kernel_regularizer=w_loss, name=name)
            ts = tf.layers.dropout(
                ts, rate=self.dropout_prob, training=self.is_training)
            return ts

        ts = inputs
        with tf.variable_scope('vgg_16'):
            with tf.variable_scope('conv1'):
                ts = conv2d(ts, 64, name='conv1_1')
                ts = conv2d(ts, 64, name='conv1_2')
                ts = pool2d(ts, name='pool1')
            with tf.variable_scope('conv2'):
                ts = conv2d(ts, 128, name='conv2_1')
                ts = conv2d(ts, 128, name='conv2_2')
                ts = pool2d(ts, name='pool2')
            with tf.variable_scope('conv3'):
                ts = conv2d(ts, 256, name='conv3_1')
                ts = conv2d(ts, 256, name='conv3_2')
                ts = conv2d(ts, 256, name='conv3_3')
                ts = pool2d(ts, name='pool3')
            with tf.variable_scope('conv4'):
                ts = conv2d(ts, 512, name='conv4_1')
                ts = conv2d(ts, 512, name='conv4_2')
                ts = conv2d(ts, 512, name='conv4_3')
                ts = pool2d(ts, name='pool4')
            with tf.variable_scope('conv5'):
                ts = conv2d(ts, 512, name='conv5_1')
                ts = conv2d(ts, 512, name='conv5_2')
                ts = conv2d(ts, 512, name='conv5_3')
                ts = pool2d(ts, name='pool5')
            # use conv2d instead of dense
            ts = conv2d(ts, 4096, kernel_size=7,
                        padding='valid', name='fc6')
            ts = dropout(ts)

            ts = conv2d(ts, 4096, kernel_size=1, name='fc7')
            ts = dropout(ts)
            # linear fc
            ts = tf.layers.conv2d(ts, self.num_class, 1,
                                  activation=None,
                                  kernel_initializer=tf.zeros_initializer(),
                                  kernel_regularizer=w_loss,
                                  name='fc8')
            pp('layer %s: units %d, gaussian 0.01 without relu and droupout'
               % ('fc8', self.num_class))
            ts = tf.squeeze(ts, [1, 2], name='fc8/squeezed')
            pp(' output shape:', ts.get_shape().as_list())

        return ts

    def get_vars_to_restore(self):

        def name_alter(name):
            return re.sub(r':\d+$', '', name) \
                .replace('kernel', 'weights').replace('bias', 'biases')

        vars_to_restore = {name_alter(var.name): var for var in tf.trainable_variables()
                           if var.name.startswith('vgg_16') and 'fc8' not in var.name}
        return vars_to_restore
