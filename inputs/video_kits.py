"""Kits for preprocessing video [..., height, width, channels]
Note: some of process functions only work for 3-D tensor [height, width, channels]
"""
from __future__ import division
import os
import numpy as np
import tensorflow as tf

from config.config_agent import FLAGS
from kits import pp


def random_crop(ts, new_size):
    """Only crop along [height, width]

    Args:
        ts (tensor): [..., height, width, in_channels]
        new_size (list): [height, width]

    Returns:
        tensor: cropped ts with reshaped [height, width]
    """
    shape = ts.get_shape().as_list()
    ts_shape = tf.concat([tf.constant(shape[:-3], dtype=tf.int32),
                          new_size, shape[-1:]], axis=0)

    pp('random crop %s' % str(new_size))
    with tf.name_scope('random_crop'):
        return tf.random_crop(ts, ts_shape)


def random_size_and_crop(ts, new_size):
    """random scale and random crop
    Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
    """
    raw_shape = ts.get_shape().as_list()
    raw_height, raw_width = raw_shape[-3:-1]
    raw_area = tf.constant(raw_height * raw_width, dtype=tf.float32)
    new_area = tf.random_uniform([], 0.08, 1.0) * raw_area
    aspect_ratio = tf.random_uniform([], 3. / 4, 4. / 3)
    crop_height = tf.round(tf.sqrt(new_area / aspect_ratio))
    crop_width = tf.round(tf.sqrt(new_area * aspect_ratio))
    # clip on crop_width, crop_height
    crop_height = tf.clip_by_value(tf.to_int32(crop_height), 0, raw_height)
    crop_width = tf.clip_by_value(tf.to_int32(crop_width), 0, raw_width)

    out_ts = crop_and_resize(ts, [crop_height, crop_width],
                             'random', new_size)
    out_ts.set_shape(raw_shape[:-3] + new_size + raw_shape[-1:])

    pp('random_size_and_crop')
    return out_ts


def random_fix_crop_with_multi_scale(ts, scale_ratios, new_size):
    # Fixed corner cropping and "Multi-scale" cropping augmentation
    # (In Xiong's Caffe)
    assert len(ts.get_shape()) == 3, 'only use for image'
    raw_shape = ts.get_shape().as_list()
    crop_pos_lst = ['left_top', 'left_bottom',
                    'mid', 'right_top', 'right_bottom']

    pp('random_fix_crop_with_multi_scale: scale_ratios %s, new_size %s' %
       (str(scale_ratios), str(new_size)))

    with tf.name_scope('random_fix_crop_with_multi_scale'):
        # random select from scale_ratios
        scale_idx = tf.random_uniform(
            [], maxval=len(scale_ratios), dtype=tf.int32)

        scale_fns = [(tf.equal(scale_idx, i),
                      lambda scale=scale: tf.constant([scale] * 2))
                     for i, scale in enumerate(scale_ratios)]
        crop_size_ts = tf.case(scale_fns, default=scale_fns[0][1])
        # randomly crop one position
        crop_idx = tf.random_uniform(
            [], maxval=len(crop_pos_lst), dtype=tf.int32)
        crop_fns = [(tf.equal(crop_idx, i),
                     lambda pos=pos: crop_and_resize(ts, crop_size_ts, pos, new_size))
                    for i, pos in enumerate(crop_pos_lst)]
        ts = tf.case(crop_fns, default=crop_fns[0][1])

        return tf.reshape(ts, new_size + raw_shape[-1:])


def crop(ts, crop_size, kind):
    """Only crop along [height, width]

    Args:
        ts (tensor): [..., height, width, in_channels]
        crop_size (list or 1-D tensor): [new_height, new_width]
        kind: way of crop

    Returns:
        tensor: cropped ts with reshaped [height, width]
    """
    raw_shape = ts.get_shape().as_list()
    height, width = raw_shape[-3:-1]
    new_height, new_width = crop_size[0], crop_size[1]
    # only allow 5 kinds of crops
    if kind == 'mid':
        H_start, W_start = [(height - new_height) // 2,
                            (width - new_width) // 2]
    elif kind == 'left_top':
        H_start, W_start = [0, 0]
    elif kind == 'left_bottom':
        H_start, W_start = [height - new_height, 0]
    elif kind == 'right_top':
        H_start, W_start = [0, width - new_width]
    elif kind == 'right_bottom':
        H_start, W_start = [height - new_height, width - new_width]

    with tf.name_scope('fix_crop'):
        # reshape to [-1, height, width, in_channels]
        ts = tf.reshape(ts, [-1] + raw_shape[-3:])
        ts_shape = ts.get_shape().as_list()
        out_ts = tf.slice(ts, [0, H_start, W_start, 0],
                          [ts_shape[0], new_height, new_width, ts_shape[-1]])
        # out_ts = ts[:, H_start:H_start + new_height,
        #             W_start:W_start + new_width, :]

        raw_shape[-3:-1] = new_height, new_width

        return tf.reshape(out_ts, raw_shape)


def crop_and_resize(ts, crop_size, crop_kind, new_size):
    """crop and resize to a image
    crop_kind: two ways
    - 'random': random_crop
    - 'mid'... fix_crop
    """
    assert len(ts.get_shape()) == 3, 'images only'
    with tf.name_scope('crop_and_resize'):
        crop_size = tf.reshape(crop_size, [2])
        if crop_kind == 'random':
            ts = random_crop(ts, crop_size)
        else:
            ts = crop(ts, crop_size, crop_kind)
        return tf.image.resize_images(ts, new_size)


def random_flip_left_right(ts, prob):
    """randomly flip horizontally

    Args:
        ts (tensor): [..., height, width, in_channels]
        prob (float): probability for flipping

    Returns:
        tensor: flip ts horizontally
    """
    raw_shape = ts.get_shape().as_list()
    with tf.name_scope('random_flip_left_right'):
        # reshape to [..., width, in_channels]
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
    # ts (a tensor): last dimension must be 3 or multiple of 3.
    num_channel = ts.get_shape().as_list()[-1]
    # subtract mean from mean file
    if 'mean_npy' not in FLAGS['input']:
        # default order of channels: BGR
        mean = np.array([104., 117., 123.])
        mean = np.tile(mean, num_channel // mean.shape[0])
    else:
        # load mean file
        mean_file = FLAGS['input']['mean_npy']
        mean_path = os.path.expanduser(mean_file)
        mean = np.load(mean_path)

    print('subtract_mean: %s' % str(mean.shape))
    with tf.name_scope('subtract_mean'):
        return ts - mean


# def whitening_per_frame(clip):
    # # clip: [depth, height, width, in_channels]
    # # return one clip
    # raise ValueError('...')
    # image_lst = tf.unpack(clip)
    # return tf.pack([tf.image.per_image_whitening(image)
    #                 for image in image_lst])


def color_jitter(ts):
    """works for color channels
    - random brightness
    - random contrast
    - random saturation
    - random hue
    Follow Inception-style:
    https://github.com/tensorflow/models/blob/master/inception/inception/image_processing.py#L183-L186
    """
    ts = tf.image.random_brightness(ts, max_delta=32. / 255.)
    ts = tf.image.random_saturation(ts, lower=0.5, upper=1.5)
    # ts = tf.image.random_hue(ts, max_delta=0.2)
    ts = tf.image.random_contrast(ts, lower=0.5, upper=1.5)

    # The random_* ops do not necessarily clamp.
    ts = tf.clip_by_value(ts, 0.0, 1.0)

    pp('color jitter (Inception-style)')
    return ts
