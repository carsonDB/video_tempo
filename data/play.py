"""
This script is only for test.
Not the part of the project.
"""

from importlib import import_module
import argparse
import threading
import numpy as np
import tensorflow as tf

num_reader = 4


def thread(idx):
    for i in range(10):
        print idx

with tf.Graph().as_default(), tf.Session() as sess:
    coord = tf.train.Coordinator()

    thread_lst = [threading.Thread(target=thread, args=(i,))
                  for i in range(num_reader)]

    for i in range(num_reader):
        tf.train.add_queue_runner(thread_lst[i])

    tf.train.start_queue_runners(sess=sess, coord=coord)

    # for i in range(num_reader):
    #     thread_lst[i].start()
