"""convert *.caffemodel to *.hdf5 (with transpose ?)

Example: python caffemodel_to_hdf5.py ./source.caffemodel ./destination.hdf5
"""
from __future__ import division
import sys
import numpy as np
import h5py
import caffe


def main():
    if len(sys.argv) != 4:
        print("python caffemodel_to_hdf5.py"
              "./*_deploy.txt ./*.caffemodel ./*.hdf5")
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
            assert(len(weights.shape) == 4)
            reshaped_weights = np.transpose(weights, [2, 3, 1, 0])
        elif l_name.startswith('fc'):
            assert(len(weights.shape) == 2)
            reshaped_weights = np.transpose(weights, [1, 0])
        else:
            raise ValueError("haven't been support: %s", l_name)

        hdf5_f.create_dataset('%s/weights' % l_name, data=reshaped_weights)
        hdf5_f.create_dataset('%s/biases' % l_name, data=biases)


if __name__ == '__main__':
    main()
