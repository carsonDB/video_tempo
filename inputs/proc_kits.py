from __future__ import division
import os
import numpy as np
import tensorflow as tf

from config.config_agent import FLAGS


# def resize_video(clip, op):
#     # clip: [depth, height, width, in_channels]
#     # return one clip
#     raise ValueError('...')
#     # is isotropic ?
#     return tf.image.resize_images(clip, *op['size'])


def random_crop(ts, size):
    """Only crop along [height, width]

    Args:
        ts (tensor): [..., height, width, in_channels]
        size (list): [height, width]

    Returns:
        tensor: cropped ts with reshaped [height, width]
    """
    shape = ts.get_shape().as_list()
    shape[-3:-1] = size
    return tf.random_crop(ts, shape)


def crop(ts, re_size, kind):
    """Only crop along [height, width]

    Args:
        ts (tensor): [..., height, width, in_channels]
        re_size (list): [re_height, re_width]
        kind: way of crop

    Returns:
        tensor: cropped ts with reshaped [height, width]
    """
    raw_shape = ts.get_shape().as_list()
    height, width = raw_shape[-3:-1]
    re_height, re_width = re_size

    # only allow 5 kinds of crops
    if kind == 'mid':
        H_start, W_start = [(height - re_height)//2, (width - re_width)//2]
    elif kind == 'left_top':
        H_start, W_start = [0, 0]
    elif kind == 'left_bottom':
        H_start, W_start = [height - re_height, 0]
    elif kind == 'right_top':
        H_start, W_start = [0, width - re_width]
    elif kind == 'right_bottom':
        H_start, W_start = [height - re_height, width - re_width]

    # reshape to [..., height, width, in_channels]
    ts = tf.reshape(ts, [-1] + raw_shape[-3:])

    out_ts = ts[:, H_start:H_start + re_height,
                W_start:W_start + re_width, :]

    raw_shape[-3:-1] = re_size
    return tf.reshape(out_ts, raw_shape)


def random_flip_left_right(ts, prob):
    """randomly flip horizontally

    Args:
        ts (tensor): [..., height, width, in_channels]
        prob (float): probability for flipping

    Returns:
        tensor: flip ts horizontally
    """
    # reshape to [..., width, in_channels]
    raw_shape = ts.get_shape().as_list()
    ts = tf.reshape(ts, [-1] + raw_shape[-2:])

    if_flip = tf.random_uniform([])
    ts = tf.cond(if_flip < prob,
                 lambda: ts[:, ::-1, :],
                 lambda: ts)
    return tf.reshape(ts, raw_shape)


# def double_flip_left_right(clip_lst, op):
#     # clip_lst: [group, ..., depth, height, width, in_channels]
#     # reshape to [..., width, in_channels]
#     raw_shape = clip_lst.get_shape().as_list()
#     reshaped = tf.reshape(clip_lst, [-1] + raw_shape[-2:])

#     mirror = reshaped[:, ::-1, :]

#     return tf.concat(0, [clip_lst, tf.reshape(mirror, raw_shape)])


# def _real_length(ts, axis):
#     used = tf.sign(tf.reduce_max(tf.abs(ts), reduction_indices=axis))
#     length = tf.reduce_sum(used, reduction_indices=(axis-1))
#     length = tf.cast(length, tf.int32)
#     return length


def subtract_mean(ts):
    # subtract mean from mean file
    if 'mean_npy' not in FLAGS['input']:
        # default order of channels: BGR
        mean = np.array([104., 117., 124.])
    else:
        # load mean file
        mean_file = FLAGS['input']['mean_npy']
        mean_path = os.path.expanduser(mean_file)
        mean = np.load(mean_path)

    return ts - mean


# def whitening_per_frame(clip):
    # # clip: [depth, height, width, in_channels]
    # # return one clip
    # raise ValueError('...')
    # image_lst = tf.unpack(clip)
    # return tf.pack([tf.image.per_image_whitening(image)
    #                 for image in image_lst])
