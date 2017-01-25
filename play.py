"""
This script is only for test.
Not the part of the project.
"""
import threading
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import raw_rnn


class A():

    def __init__(self):
        self.lr = 2.3
        self.a = tf.Variable(0, name='a',
                             trainable=False,
                             dtype=tf.float32)
        self.a = tf.py_func(self._op, [self.a], tf.float32)

    def _op(self, a):
        # import ipdb; ipdb.set_trace()
        return (a + self.lr).astype(np.float32)


with tf.Graph().as_default():
    sess = tf.Session()
    a = A()
    # initialize variables
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)
    print(sess.run(a.a))
