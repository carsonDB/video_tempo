import numpy as np
import tensorflow as tf

# input local file
from config.config_agent import FLAGS
import model_proc

"""
args:
    clip:
        * video: [depth, height, width, in_channels]
        * seq_video: [num_step, depth, height, width, in_channels]
    group of clips:
        * videos: [group, depth, height, width, in_channels]
        * seq_videos: [group, num_step, depth, height, width, in_channels]
return:
    list of tensors
"""

THIS = {}


# deprecated !!!
def wrapper_per_clip(func):
    """
    Input:
        * clip
        * seq_clip
    inner-input:
        * clip
    inner-output:
        * clip
    Output:
        * clip
        * seq_clip
    """
    def inner(clip, *args, **kwargs):
        rank = len(clip.get_shape())
        # a clips: [depth, height, width, in_channels]
        if rank == 4:
            return func(clip, *args, **kwargs)
        # sequence of clips: [steps, depth, height, width, in_channels]
        elif rank == 5:
            seq_re = []
            seq_lst = tf.unpack(clip)
            for c in seq_lst:
                seq_re.append(func(c, *args, **kwargs))
            return tf.pack(seq_re)
        else:
            raise ValueError('video proc input shape illegal: rank==%d', rank)

    return inner


# @wrapper_per_lst...
def resize_video(clip, op):
    # clip: [depth, height, width, in_channels]
    # return one clip
    raise ValueError('...')
    return tf.image.resize_images(clip, *op['size'])


def random_crop(clip_lst, op):
    # clip_lst: [group, ..., depth, height, width, in_channels]
    shape = [dim.value for dim in clip_lst.get_shape()]
    shape[-3:-1] = op['size']
    return tf.random_crop(clip_lst, shape)


def central_crop(clip, *args, **kwargs):
    # clip: [depth, height, width, in_channels]
    # return one clip
    raise ValueError('...')
    image_lst = tf.unpack(clip)
    return tf.pack([tf.image.central_crop(image, *args, **kwargs)
                    for image in image_lst])


def crop(clip_lst, op):
    # clip_lst: [group, ..., depth, height, width, in_channels]
    height, width = clip_lst.get_shape().as_list()[-3:-1]
    re_height, re_width = op['size']

    if op['pos'] == 'all':
        pos_lst = ['lt', 'lb', 'rt', 'rb', 'mid']
    else:
        pos_lst = [op['pos']]

    start_pos_lst = []
    if 'mid' in pos_lst:
        start_pos_lst.append([(height - re_height)/2, (width - re_width)/2])
    if 'lt' in pos_lst:
        start_pos_lst.append([0, 0])
    if 'lb' in pos_lst:
        start_pos_lst.append([height - re_height, 0])
    if 'rt' in pos_lst:
        start_pos_lst.append([0, width - re_width])
    if 'rb' in pos_lst:
        start_pos_lst.append([height - re_height, width - re_width])

    # reshape to [..., height, width, in_channels]
    raw_shape = clip_lst.get_shape().as_list()
    clip_lst = tf.reshape(clip_lst, [-1] + raw_shape[-3:])

    out_lst = []
    for H_start, W_start in start_pos_lst:
        out_lst.append(clip_lst[:, H_start:H_start + re_height,
                       W_start:W_start + re_width, :])

    raw_shape[-3:-1] = op['size']
    return tf.reshape(tf.concat(0, out_lst), [-1] + raw_shape[1:])


def random_flip_left_right(clip_lst, op):
    # clip_lst: [group, ..., depth, height, width, in_channels]
    # reshape to [..., width, in_channels]
    raw_shape = clip_lst.get_shape().as_list()
    clip_lst = tf.reshape(clip_lst, [-1] + raw_shape[-2:])

    if_flip = tf.random_uniform([])
    clip_lst = tf.cond(if_flip < op['prob'],
                       lambda: clip_lst[:, ::-1, :],
                       lambda: clip_lst)
    return tf.reshape(clip_lst, raw_shape)


def double_flip_left_right(clip_lst, op):
    # clip_lst: [group, ..., depth, height, width, in_channels]
    # reshape to [..., width, in_channels]
    raw_shape = clip_lst.get_shape().as_list()
    reshaped = tf.reshape(clip_lst, [-1] + raw_shape[-2:])

    mirror = reshaped[:, ::-1, :]

    return tf.concat(0, [clip_lst, tf.reshape(mirror, raw_shape)])


# def _real_length(ts, axis):
#     used = tf.sign(tf.reduce_max(tf.abs(ts), reduction_indices=axis))
#     length = tf.reduce_sum(used, reduction_indices=(axis-1))
#     length = tf.cast(length, tf.int32)
#     return length


def subtract_mean(clip_lst, op):
    # subtract mean from mean file
    if 'mean' is not THIS:
        # load mean file
        mean_file = FLAGS['input']['mean_file']
        THIS['mean'] = mean = np.load(mean_file)
    else:
        mean = THIS['mean']

    # subtract actual data if time_step exists
    if clip_lst.get_shape().ndims > 5:
        # time_step is at [-5]
        raw_shape = clip_lst.get_shape().as_list()
        reshaped = tf.reshape(clip_lst, [-1] + raw_shape[-5:])
        mid_shape = reshaped.get_shape().as_list()
        reshaped = tf.reshape(reshaped, mid_shape[:2] + [-1])
        used = tf.sign(tf.reduce_max(tf.abs(reshaped), reduction_indices=2))
        used = tf.reshape(used, used.get_shape().as_list()+[1])
        return clip_lst - tf.reshape(used*mean.ravel(), raw_shape)
    return clip_lst - mean


def whitening_per_frame(clip):
    # clip: [depth, height, width, in_channels]
    # return one clip
    raise ValueError('...')
    image_lst = tf.unpack(clip)
    return tf.pack([tf.image.per_image_whitening(image)
                    for image in image_lst])


def model_process(clip, *args, **kwargs):
    # clip(must a tensor): [depth, height, width, in_channels]
    # return one clip
    # model should be fed with [batch_size, ...]
    raise ValueError('...')

    rank = len(clip.get_shape())

    # list of clips: [depth, height, width, in_channels]
    if rank == 4:
        return model_proc.feature_extract(tf.pack([clip]), *args, **kwargs)

    # sequence of clips: [steps, depth, height, width, in_channels]
    elif rank == 5:
        return model_proc.feature_extract(clip, *args, **kwargs)
    else:
        raise ValueError('video proc input shape illegal')


TOOLS = {
    "resize": resize_video,
    "subtract_mean": subtract_mean,
    "random_crop": random_crop,
    "crop": crop,
    "random_flip_left_right": random_flip_left_right,
    "double_flip_left_right": double_flip_left_right,
    "whitening": whitening_per_frame,
    "__model__": model_process
}


def proc(example, sess, group_head=False):
    """ preprocess actually based on frames
    input -> output:
        * 1 -> 1
        * 1 -> n
        * n -> 1 (group_head)
    """
    PREPROC = FLAGS['preproc']

    # first dimension for multi-generating
    example_lst = tf.pack([example])

    for op in PREPROC:
        if op['name'] in TOOLS:
            example_lst = TOOLS[op['name']](example_lst, op)

    # output
    if group_head is False:
        if example_lst.get_shape()[0].value == 1:
            out, is_multi = example_lst[0], False
        else:
            out, is_multi = example_lst, True
    else:
        # mainly for test
        out, is_multi = example_lst, False

    return out, is_multi
