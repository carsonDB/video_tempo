"""
This script is only for test.
Not the part of the project.
"""

from importlib import import_module
import threading
import numpy as np
import tensorflow as tf


a = tf.get_variable('a', shape=[2, 5],
                    initializer=tf.truncated_normal_initializer())
b = tf.get_variable('b', shape=[5, 9],
                    initializer=tf.truncated_normal_initializer())

sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# c = sess.run(tf.matmul(a, b))

print len(a.get_shape())