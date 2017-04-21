"""Build c3d model (8 layers of Conv-3D)
"""
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import re

from config.config_agent import FLAGS, VARS
from kits import variable_on_cpu, variable_with_weight_decay, pp
from models.model_proto import Model_proto


class Model(Model_proto):

    def __init__(self):
        super(Model, self).__init__()
        self.is_training = (VARS['mode'] == 'train')
        self.dropout_prob = self.model_config['dropout']
        self.graph = self.model_config['graph']

    def infer(self, inputs):
        # inputs: [b, h, w, s * c]
        inputs_shape = inputs.get_shape().as_list()
        pp('inputs shape: ', inputs_shape)
        self.input_batch_size = inputs_shape[0]
        frm_step = inputs_shape[-1] // 3

        # transform -> [b, s, h, w, c]
        ts = tf.reshape(inputs, inputs_shape[:3] + [frm_step, 3])
        ts = tf.transpose(ts, [0, 3, 1, 2, 4])
        pp('transform inputs shape to: ', ts.get_shape().as_list())

        # test: transform
        # tf.summary.image()

        def w_loss(ts, weight_decay=0.0005):
            return tf.nn.l2_loss(ts) * weight_decay

        def reshape_rank2(ts):
            # shape [-1, last dimension]
            ts_shape = ts.get_shape().as_list()
            return tf.reshape(ts, [ts_shape[0], -1])

        with tf.variable_scope('c3d'):
            for ly in self.graph:
                if ly['name'].startswith('conv'):
                    ts = tf.layers.conv3d(ts, ly['num_kernel'], [3, 3, 3], padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.truncated_normal_initializer(
                                              stddev=0.01),
                                          kernel_regularizer=w_loss,
                                          name=ly['name'])
                    pp('layer %s: num_kernel %d, stride [3, 3, 3],'
                       ' gaussian 0.01, l2_loss, with relu' % (ly['name'], ly['num_kernel']))
                    pp(ts)
                elif ly['name'].startswith('pool'):
                    ts = tf.layers.max_pooling3d(ts, ly['kernel_size'],
                                                 ly['stride'],
                                                 padding='same',
                                                 name=ly['name'])
                    pp('layer %s: kernel_size %s, stride %s'
                       % (ly['name'], str(ly['kernel_size']), str(ly['stride'])))
                    pp(ts)
                elif ly['name'].startswith('fc'):
                    # reshape to rank 2
                    ts_shape = ts.get_shape().as_list()
                    if len(ts.get_shape().as_list()) > 2:
                        ts = reshape_rank2(ts)

                    ts = tf.layers.dense(ts, ly['units'],
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.truncated_normal_initializer(
                                             stddev=0.005),
                                         bias_initializer=tf.ones_initializer(),
                                         kernel_regularizer=w_loss,
                                         name=ly['name'])
                    ts = tf.layers.dropout(ts, rate=self.dropout_prob,
                                           training=self.is_training)
                    pp('layer %s: units %d, gaussian 0.005, l2_loss,'
                       ' with relu and droupout %.1f(%r)' %
                       (ly['name'], ly['units'], self.dropout_prob, self.is_training))
                    pp(ts)
                else:
                    raise ValueError('no such layer %s' % ly['name'])

            # linear fc
            ts = tf.layers.dense(ts, 101, activation=None,
                                 kernel_initializer=tf.truncated_normal_initializer(
                                     stddev=0.01),
                                 kernel_regularizer=w_loss,
                                 name='fc8')
            pp('layer %s: units 101, gaussian 0.01 without relu and droupout'
               % ('fc8'))
            pp(ts)

        return ts

    def get_vars_to_restore(self):
        # ckpt from tencent
        # model_to_ckpt = {
        #     'conv1a/kernel': 'wc1',
        #     'conv2a/kernel': 'wc2',
        #     'conv3a/kernel': 'wc3a',
        #     'conv3b/kernel': 'wc3b',
        #     'conv4a/kernel': 'wc4a',
        #     'conv4b/kernel': 'wc4b',
        #     'conv5a/kernel': 'wc5a',
        #     'conv5b/kernel': 'wc5b',
        #     'conv1a/bias': 'bc1',
        #     'conv2a/bias': 'bc2',
        #     'conv3a/bias': 'bc3a',
        #     'conv3b/bias': 'bc3b',
        #     'conv4a/bias': 'bc4a',
        #     'conv4b/bias': 'bc4b',
        #     'conv5a/bias': 'bc5a',
        #     'conv5b/bias': 'bc5b',
        #     'fc6/kernel': 'wd1',
        #     'fc7/kernel': 'wd2',
        #     'fc8/kernel': 'wout',
        #     'fc6/bias': 'bd1',
        #     'fc7/bias': 'bd2',
        #     'fc8/bias': 'bout',
        # }
        # temp from caffe
        model_to_ckpt = {
            'conv1a/kernel': 'conv1a/weights',
            'conv2a/kernel': 'conv2a/weights',
            'conv3a/kernel': 'conv3a/weights',
            'conv3b/kernel': 'conv3b/weights',
            'conv4a/kernel': 'conv4a/weights',
            'conv4b/kernel': 'conv4b/weights',
            'conv5a/kernel': 'conv5a/weights',
            'conv5b/kernel': 'conv5b/weights',
            'conv1a/bias': 'conv1a/biases',
            'conv2a/bias': 'conv2a/biases',
            'conv3a/bias': 'conv3a/biases',
            'conv3b/bias': 'conv3b/biases',
            'conv4a/bias': 'conv4a/biases',
            'conv4b/bias': 'conv4b/biases',
            'conv5a/bias': 'conv5a/biases',
            'conv5b/bias': 'conv5b/biases',
            # 'fc6/kernel': 'fc6-1/weights',
            # 'fc7/kernel': 'fc7-1/weights',
            # 'fc8/kernel': 'fc8/weights',
            # 'fc6/bias': 'fc6-1/biases',
            # 'fc7/bias': 'fc7-1/biases',
            # 'fc8/bias': 'fc8/biases',
        }

        def name_alter(name):
            # 'c3d/conv1a/kernel:0' -> 'conv1a/kernel'
            name = re.sub(r':\d+$', '', name).replace('c3d/', '')
            local = model_to_ckpt[name]
            # return 'var_name/' + local
            # temp
            return local

        # vars_to_restore = {name_alter(var.name): var for var in tf.trainable_variables()
        #                    if var.name.startswith('c3d')}
        vars_to_restore = {name_alter(var.name): var for var in tf.trainable_variables()
                           if var.name.startswith('c3d') and not 'fc' in var.name}
        return vars_to_restore
