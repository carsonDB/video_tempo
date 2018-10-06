"""Build vgg model
"""
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
import numpy as np

from config.config_agent import FLAGS, VARS
from kits import variable_on_cpu, variable_with_weight_decay, pp
from models.model_proto import Model_proto

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12


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


class Model(Model_proto):

    def __init__(self):
        super(Model, self).__init__()
        self.is_training = (VARS['mode'] == 'train')
        self.dropout_prob = self.model_config['dropout']
        self.feedback = self.model_config['feedback']
        self.layers = {}

    def infer(self, inputs):
        input_shape = inputs.get_shape().as_list()
        self.input_batch_size = input_shape[0]
        self.num_step = self.model_config['num_step']
        input_lst = tf.split(inputs, self.num_step, -1)

        logits_lst = []
        with tf.variable_scope('vgg_16'):
            for ts in input_lst:
                logits = self._infer_body(ts)
                logits_lst.append(logits)
                if self.feedback is True:
                    # add feedback_loss
                    self._add_pred_loss()
                    # update feedback
                    self._reverse_body()
                tf.get_variable_scope().reuse_variables()

        ave_logits = tf.reduce_mean(logits_lst, axis=0)
        # linear fc
        final_logits = tf.layers.conv2d(ave_logits, self.num_class, 1,
                                        activation=None,
                                        kernel_initializer=tf.zeros_initializer(),
                                        kernel_regularizer=w_loss,
                                        name='fc8')
        pp('layer %s: units %d, gaussian 0.01 without relu and droupout'
           % ('fc8', self.num_class))
        final_logits = tf.squeeze(final_logits, [1, 2], name='fc8/squeezed')
        pp(' output shape:', ts.get_shape().as_list())

        return final_logits

    def dropout(self, ts):
        return tf.layers.dropout(
            ts, rate=self.dropout_prob, training=self.is_training)

    def dense(self, ts, units, name):
        ts = tf.layers.dense(ts, units, activation=tf.nn.relu,
                             kernel_regularizer=w_loss, name=name)
        ts = tf.layers.dropout(
            ts, rate=self.dropout_prob, training=self.is_training)
        return ts

    def _infer_body(self, ts):
        with tf.variable_scope('conv1'):
            self.layers['conv1_1'] = ts = conv2d(ts, 64, name='conv1_1')
            self.layers['conv1_2'] = ts = conv2d(ts, 64, name='conv1_2')
            ts = pool2d(ts, name='pool1')
        with tf.variable_scope('conv2'):
            self.layers['conv2_1'] = ts = conv2d(ts, 128, name='conv2_1')
            self.layers['conv2_2'] = ts = conv2d(ts, 128, name='conv2_2')
            ts = pool2d(ts, name='pool2')
        with tf.variable_scope('conv3'):
            self.layers['conv3_1'] = ts = conv2d(ts, 256, name='conv3_1')
            self.layers['conv3_2'] = ts = conv2d(ts, 256, name='conv3_2')
            self.layers['conv3_3'] = ts = conv2d(ts, 256, name='conv3_3')
            ts = pool2d(ts, name='pool3')
        with tf.variable_scope('conv4'):
            self.layers['conv4_1'] = ts = conv2d(ts, 512, name='conv4_1')
            self.layers['conv4_2'] = ts = conv2d(ts, 512, name='conv4_2')
            self.layers['conv4_3'] = ts = conv2d(ts, 512, name='conv4_3')
            ts = pool2d(ts, name='pool4')
        with tf.variable_scope('conv5'):
            self.layers['conv5_1'] = ts = conv2d(ts, 512, name='conv5_1')
            self.layers['conv5_2'] = ts = conv2d(ts, 512, name='conv5_2')
            self.layers['conv5_3'] = ts = conv2d(ts, 512, name='conv5_3')
            ts = pool2d(ts, name='pool5')
        # use conv2d instead of dense
        self.layers['fc6'] = ts = conv2d(ts, 4096, kernel_size=7,
                                         padding='valid', name='fc6')
        ts = self.dropout(ts)

        self.layers['fc7'] = ts = conv2d(ts, 4096, kernel_size=1, name='fc7')
        ts = self.dropout(ts)

        return ts

    def _depthwise_conv2d(self, inputs, kernels):
        # conv for each example.
        kernel_lst = tf.split(
            axis=0, num_or_size_splits=self.input_batch_size, value=kernels)
        input_lst = tf.split(
            axis=0, num_or_size_splits=self.input_batch_size, value=inputs)

        # Transform inputs.
        transformed = []
        for kernel, input in zip(kernel_lst, input_lst):
            kernel = tf.squeeze(kernel)
            if len(kernel.get_shape()) == 3:
                kernel = tf.expand_dims(kernel, -1)
            transformed.append(
                tf.nn.depthwise_conv2d(input, kernel, [1, 1, 1, 1], 'SAME'))
        transformed = tf.concat(axis=0, values=transformed)

        return transformed

    def _fc_pred_layer(self, top_ts, target_ts, kernel=5, channel=64, name=None):
        # constant init [kernel,kernl,channel] with middle 1
        np_init = np.zeros([kernel, kernel, channel, 1], dtype=np.float32)
        np_init[kernel//2, kernel//2, :, :] = 1.0
        init_ts = tf.constant(np_init, dtype=tf.float32)

        top_ts = tf.reshape(top_ts, [self.input_batch_size, -1])
        trans_ts = tf.layers.dense(top_ts, kernel**2*channel, activation=None,
                                   # kernel_initializer=tf.zeros_initializer(),
                                   kernel_regularizer=w_loss, name=name)
        trans_ts = tf.reshape(
            trans_ts, [self.input_batch_size, kernel, kernel, channel, 1])
        trans_ts += init_ts

        trans_ts = tf.nn.relu(trans_ts - RELU_SHIFT) + RELU_SHIFT
        norm_factor = tf.reduce_sum(trans_ts, [1, 2], keep_dims=True)
        trans_ts /= norm_factor
        pred_ts = self._depthwise_conv2d(tf.stop_gradient(target_ts), trans_ts)
        # pred_ts = self._depthwise_conv2d(target_ts, trans_ts)

        return pred_ts

    def _conv_pred_layer(self, top_ts, target_ts, name):
        _, target_h, target_w, target_channel = target_ts.get_shape().as_list()
        top_ts = tf.image.resize_bilinear(top_ts, [target_h, target_w])
        pred_ts = conv2d(top_ts, target_channel, name=name) + target_ts
        return pred_ts

    def _reverse_body(self):
        # || (self.conv1_1<t> * self.res1_1) - self.conv1_1<t+1> ||

        self.layers['pred1_1'] = self._fc_pred_layer(
            self.layers['fc6'], self.layers['conv1_1'], name='trans1_1')
        self.layers['prev_conv1_1'] = self.layers['conv1_1']
        # for i in range(4, 1, -1):
        #     input_ts = self.layers['conv%d_1' % i]
        #     target_ts = self.layers['conv%d_1' % (i-1)]
        #     self.layers['pred%d_1' % (i-1)] = self._conv_pred_layer(
        #         input_ts, target_ts, name='trans%d_1' % (i-1))

    def _add_pred_loss(self):
        # || (self.conv1_1<t> * self.res1_1) - self.conv1_1<t+1> ||
        if 'pred1_1' not in self.layers:
            return

        def _cal_pred_loss(target_ts, pred_ts, mask):
            loss = tf.nn.l2_loss(mask * (tf.stop_gradient(target_ts) - pred_ts))
            loss = tf.reduce_mean(loss / tf.reduce_sum(loss))
            return loss

        # maybe norm by variance
        # pred_loss = tf.nn.l2_loss(self.conv1_1_ts - self.pred1_1_ts)
        for i in range(1, 2):
            mask = tf.cast((self.layers['prev_conv%d_1' % i] - self.layers['conv%d_1' % i]) != 0,
                           tf.float32)
            pred_loss = _cal_pred_loss(self.layers['conv%d_1' % i],
                                       self.layers['pred%d_1' % i],
                                       mask)
            tf.add_to_collection('losses', pred_loss * 10)

    def get_vars_to_restore(self):
        exclude_snippet = ['fc8', 'trans']

        def name_alter(name):
            return re.sub(r':\d+$', '', name) \
                .replace('kernel', 'weights').replace('bias', 'biases')

        def is_exist(name, exclude_snippet=exclude_snippet):
            for n in exclude_snippet:
                if n in name:
                    return True

        vars_to_restore = {name_alter(var.name): var for var in tf.trainable_variables()
                           if var.name.startswith('vgg_16') and not is_exist(var.name)}
        return vars_to_restore
