"""
Debug mainly on input and preprocess
"""

import tensorflow as tf
import cv2
import numpy as np

# import local file
from config import config_agent
from config.config_agent import FLAGS
from input import input_agent
from input import video_proc


config_agent.init_FLAGS('train')

def check_random_crop(row_np, cropped_np):
    # check random_crop
    for y in range(256 - 112):
        for x in range(340 - 112):
            if np.array_equal(cropped_np, raw_np[:, y:y+112, x:x+112, :]):
                print 'have found, y: %d, x: %d' % (y, x)
                break

# create a session and a coordinator
sess = tf.Session(config=tf.ConfigProto(
    log_device_placement=True))
coord = tf.train.Coordinator()

# raw data
inputs, labels, readers = input_agent.read(sess, coord)
raw_lst = tf.unpack(inputs)
label_lst = tf.unpack(labels)
# test_crop
test_lst = [video_proc.test_crop(raw_video, [112, 112])
            for raw_video in raw_lst]
# subtract mean
# mean_lst = [video_proc.subtract_mean(raw_video)
#             for raw_video in raw_lst]
# croppped data
# cropped_lst = [video_proc.random_crop(raw_video, [16, 112, 112, 3])
#                 for raw_video in raw_lst]
# flip_left_right
# flip_lst = [video_proc.random_flip_left_right(cropped_video, 0.5)
#               for cropped_video in cropped_lst]
# central_crop
# test_video = video_proc.test_crop(raw_video)

flip_count = 0
no_flip_count = 0
# raw video
for i in range(30):
    print 'step: %d' % i
    raw_np, test_np, label_id = sess.run([raw_lst[i],
                                          test_lst[i],
                                          # mean_lst[i],
                                          # cropped_lst[i],
                                          # flip_lst[i],
                                          label_lst[i]])
    print raw_np.shape
    # print 'check_sum: %f' % np.sum(raw_np - np.mean(raw_np))
    # print 'mean_sum: %f' % np.sum(mean_np)
    # print cropped_np.shape
    # print flip_np.shape
    for j, img in enumerate(raw_np):
        cv2.imwrite('./other/class%d_img%d.jpg' % (label_id, j), img)

    # check random_crop
    check_random_crop(raw_np, test_np)
    # check flip
    # if np.array_equal(cropped_np, flip_np):
    #     no_flip_count += 1
    # elif np.array_equal(cropped_np, flip_np[:, :, ::-1, :]):
    #     flip_count += 1


import pdb; pdb.set_trace()  # breakpoint 1951c376 //


coord.request_stop()
coord.join(readers)
print('training process ends normally')
