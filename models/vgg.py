"""Build vgg model
"""
from __future__ import division
import h5py
import numpy as np
import tensorflow as tf

from config.config_agent import FLAGS, VARS
from kits import variable_on_cpu, variable_with_weight_decay
from models.cnn import Model as CNN_proto


class Model(CNN_proto):
    def __init__(self):
        super(Model, self).__init__()
        self.weights_hdf5 = FLAGS["weights_hdf5"]
        self.vars_name_map = (
            # (VGG_CNN_M_2048.caffemodel, self model)
            ("conv1", "conv2d_1"),
            ("conv2", "conv2d_2"),
            ("conv3", "conv2d_3"),
            ("conv4", "conv2d_4"),
            ("conv5", "conv2d_5"),
            ("fc6", "local_6"),
            ("fc7", "local_7"),
            ("fc8", "softmax_linear"),
        )

    def model_init(self, scope=''):
        import ipdb; ipdb.set_trace()  # breakpoint 70d48e93 //
        with h5py.File(self.weights_hdf5, 'r') as f:
            # initialize variables with vgg pretrained model
            for l0, l1 in self.vars_name_map:
                for sub_name in ['weights', 'biases']:
                    npy_var = np.array(f[l0][sub_name])
                    tf_var = tf.get_variable('%s/%s/%s' % (scope, l1, sub_name))
                    # assign hdf5_weights to tf_vars
                    tf_var.assign(npy_var)

    # def loss(self):
    #     pass

    # def eval(self):
    #     pass