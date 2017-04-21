"""convert *.caffemodel to *.hdf5 (tensorflow weights format)

Example: python caffemodel_to_hdf5.py ./source.caffemodel ./destination.hdf5
"""
from __future__ import division
import sys
import numpy as np
import h5py
import caffe


def convert_by_caffe_pb2():
    pass


def convert_by_pycaffe():
    if len(sys.argv) != 4:
        print("python caffemodel_to_hdf5.py"
              " ./*_deploy.txt ./*.caffemodel ./*.hdf5")
        sys.exit()

    caffe.set_mode_gpu()
    model_def = sys.argv[1]
    model_weights = sys.argv[2]
    model_hdf5 = sys.argv[3]
    hdf5_f = h5py.File(model_hdf5, 'w')

    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't dropout)

    for l_name, param in net.params.iteritems():
        weights = param[0].data
        biases = param[1].data
        if l_name.startswith('conv'):

            if len(weights.shape) == 4:
                reshaped_weights = np.transpose(weights, [2, 3, 1, 0])
            elif len(weights.shape) == 5:
                reshaped_weights = np.transpose(weights, [2, 3, 4, 1, 0])
            else:
                raise ValueError('weights shape should be 4 or 5, not %d'
                                 % len(weights.shape))
        elif l_name.startswith('fc'):
            assert(len(weights.shape) == 2)
            # check input blobs shape rank 2 or 4, 5
            link_prev, link_next, key = net.blobs._OrderedDict__map[l_name]
            input_blob_shape = net.blobs[link_prev[2]].data.shape
            if len(input_blob_shape) != 2:
                out_channel = weights.shape[0]
                # weights in_channel: [C, H, W, ...] -> [H, W, ..., C]
                weights = weights.reshape(
                    [out_channel, input_blob_shape[1], -1])
                print('first fc layer special care: %s' % str(weights.shape))
                reshaped_weights = np.transpose(weights, [2, 1, 0]).reshape(
                    [-1, out_channel])
            else:
                reshaped_weights = np.transpose(weights, [1, 0])
        else:
            raise ValueError("haven't been support: %s", l_name)

        hdf5_f.create_dataset('%s/weights' % l_name, data=reshaped_weights)
        print('save: %s/weights' % l_name, reshaped_weights.shape)
        hdf5_f.create_dataset('%s/biases' % l_name, data=biases)
        print('save: %s/biases' % l_name, biases.shape)


if __name__ == '__main__':
    # make sure you choose the right version of caffe
    raw_input('> please check caffe version: %s\npress any key to continue...'
              % caffe.__file__)
    convert_by_pycaffe()
