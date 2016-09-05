"""
This script only tests (CNN):
    + inference part
    + loss
    - without gradient descend
"""

import tensorflow as tf
import numpy as np

# import local files
from config.config_agent import FLAGS
from config import config_agent
from input import input_agent
from model import cnn

# Compare losses of first several iterations.

def run_my_model():
    # My model only runs once.
    input_agent.init_FLAGS('train')
    train_dir = FLAGS['train_dir']

    with tf.Graph().as_default():
        sess = tf.Session()
        coord = tf.train.Coordinator()
        global_step = tf.Variable(0, trainable=False)
        # FLAGS['global_step'] = global_step

        inputs, labels, readers = input_agent.read(sess, coord)
        

