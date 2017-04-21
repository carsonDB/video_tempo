"""
This script is only for test.
Not the part of the project.
"""
import numpy as np
import tensorflow as tf


class A():

    def __init__(self):
        self.lr = 2.3

        a = self.a = tf.random_uniform([2, 1])
        self.out = self.bn(a, 'a_scope')
        with tf.variable_scope('a_scope', reuse=True):
            self.moving_mean = tf.get_variable(
                'BatchNorm/moving_mean', shape=[1])

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.moving_mean = tf.identity(self.moving_mean)

    def bn(self, ts, scope):
        is_training = True
        # is_training = (VARS['mode'] == 'train')
        with tf.variable_scope(scope):
            return tf.layers.batch_normalization(ts, training=is_training,
                                                 name='BatchNorm')


with tf.Graph().as_default():
    sess = tf.Session()
    a = A()
    # initialize variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(10):
        print(sess.run([a.a, a.out, a.moving_mean]))
        print('\n')
