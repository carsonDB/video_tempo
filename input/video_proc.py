import tensorflow as tf

# input local file
from config.config_agent import FLAGS
import model_proc

"""
args:
    clip:
        * video: [depth, height, width, in_channels]
        * seq_video: [num_step, depth, height, width, in_channels]

return:
    list of tensors
"""


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
            raise ValueError('video proc input shape illegal')

    return inner


@wrapper_per_clip
def resize_video(clip, *args, **kwargs):
    # clip: [depth, height, width, in_channels]
    # return one clip
    return tf.image.resize_images(clip, *args, **kwargs)


@wrapper_per_clip
def random_crop(clip, re_size):
    # clip: [depth, height, width, in_channels]
    # return one clip
    return tf.random_crop(clip, re_size)


@wrapper_per_clip
def central_crop(clip, *args, **kwargs):
    # clip: [depth, height, width, in_channels]
    # return one clip
    image_lst = tf.unpack(clip)
    return tf.pack([tf.image.central_crop(image, *args, **kwargs)
                    for image in image_lst])


@wrapper_per_clip
def random_flip_left_right(clip, prob):
    # clip: [depth, height, width, in_channels]
    # return one clip
    image_lst = tf.unpack(clip)
    if_flip = tf.random_uniform([])

    image_lst = tf.cond(if_flip < prob,
                        lambda: [tf.image.flip_left_right(image)
                                 for image in image_lst],
                        lambda: image_lst)
    return tf.pack(image_lst)


@wrapper_per_clip
def whitening_per_frame(clip):
    # clip: [depth, height, width, in_channels]
    # return one clip
    image_lst = tf.unpack(clip)
    return tf.pack([tf.image.per_image_whitening(image)
                    for image in image_lst])


def model_process(clip, *args, **kwargs):
    # clip(must a tensor): [depth, height, width, in_channels]
    # return one clip
    # model should be fed with [batch_size, ...]

    rank = len(clip.get_shape())

    # list of clips: [depth, height, width, in_channels]
    if rank == 4:
        return model_proc.feature_extract(tf.pack([clip]), *args, **kwargs)

    # sequence of clips: [steps, depth, height, width, in_channels]
    elif rank == 5:
        return model_proc.feature_extract(clip, *args, **kwargs)
    else:
        raise ValueError('video proc input shape illegal')


def proc(example, sess):
    # preprocess actually based on frames
    PREPROC = FLAGS['input']['preproc']

    # input: a tensor or list of tensors
    # return: list of tensor
    for op in PREPROC:

        if op['name'] == 'resize':
            example = resize_video(example, *op['size'])

        elif op['name'] == 'random_crop':
            example = random_crop(example, op['size'])

        elif op['name'] == 'flip_left_right':
            example = random_flip_left_right(example, op['prob'])

        elif op['name'] == 'whitening':
            example = whitening_per_frame(example)

        elif op['name'] == '__model__':
            # pretrain model for feature extractor
            example = model_process(example, op, sess)

    return example
