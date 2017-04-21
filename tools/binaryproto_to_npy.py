"""convert *.binaryproto to *.npy (with transpos)

Example: python binaryproto_to_npy.py ./source-path ./destination-path
"""
import caffe
import numpy as np
import sys


if len(sys.argv) != 3:
    print("python binaryproto_to_npy.py ./source-path ./destination-path")
    sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
data = open(sys.argv[1], 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
out = arr[0]
assert(len(out.shape) == 3)
# [channel, height, width] -> [height, width, channel]
reshaped_out = np.transpose(out, [1, 2, 0])
np.save(sys.argv[2], reshaped_out)
