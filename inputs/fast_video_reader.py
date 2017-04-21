raise ValueError('wrong input')
"""spatial_reader for frames (pure tensorflow API)
Raw_data format: every video consists of frames.
Output_data format:
    * train: frames randomly cutted from frames(video).
"""
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf

from config.config_agent import FLAGS, VARS
from inputs.input_proto import Input_proto
from inputs import video_kits as kits
from kits import pp


def read_rgb_randomly(csv_path, raw_size, num=1):
    """randomly slice from a video(consists of frames) with continuous frames

    Args:
        csv_path (string): absPath of csv (list).
        num (int): number of sliced frames (continuous).

    Returns:
        frames: [num_step, height, width, channel(3)].
        label: [].
    """
    csv_queue = tf.train.string_input_producer([csv_path])
    reader = tf.TextLineReader()
    _, value = reader.read(csv_queue)
    video_dir, duration, label = tf.decode_csv(value,
                                               record_defaults=(
                                                   [''], [0], [-1]),
                                               field_delim=' ')
    random_idx = tf.random_uniform([], minval=0, maxval=(duration - num),
                                   dtype=tf.int32)
    idx_range = tf.range(random_idx, limit=(random_idx + num)) + 1
    img_lst = video_dir + '/image_' + \
        tf.as_string(idx_range, width=4, fill='0') + '.jpg'

    img_name_queue = tf.FIFOQueue(32, dtypes=tf.string, shapes=[])
    enqueue_op = img_name_queue.enqueue_many(img_lst)
    VARS['queue_runners'].append(
        tf.train.QueueRunner(img_name_queue, [enqueue_op]))

    file_reader = tf.WholeFileReader()
    _, value = file_reader.read(img_name_queue)
    image = tf.image.decode_image(value, channels=3)
    image = tf.reshape(image, raw_size)[:, :, ::-1]  # rgb -> bgr
    images = tf.train.batch([image], batch_size=num)

    return images, label


class Reader(Input_proto):
    """read frames of videos
    """

    def __init__(self):
        super(Reader, self).__init__()
        self.raw_size = self.INPUT['raw_size']
        self.seq_len = self.INPUT['seq_len']
        self.num_per_video = (FLAGS['num_per_video']
                              if 'num_per_video' in FLAGS
                              else None)
        self.data_path = os.path.expanduser(FLAGS['data_path'])

    def get_data(self):
        clip, label = read_rgb_randomly(self.data_path,
                                        self.raw_size, num=1)
        # temp
        clip = tf.reshape(clip, self.raw_size)

        self.raw_inputs = {'X': clip, 'Y': label}
        # name(frame path): scalar
        if VARS['mode'] == 'test':
            self.raw_inputs['name'] = tf.placeholder(tf.string, [])

        raw_X = self.raw_inputs['X']
        # clip: uint8 -> float32
        ts = tf.to_float(raw_X)

        # subtract mean: [..., height, width, channel]
        ts = kits.subtract_mean(ts)

        # random or central crop: [height, width]
        crop_size = self.example_size[-3:-1]
        if self.mode == 'train':
            ts = kits.random_crop(ts, self.example_size[:2])
            print('train: random_crop')
            # ts = kits.random_fix_crop_with_multi_scale(ts,
            #                                            [256, 224, 192, 168],
            #                                            crop_size)
            # print('random_fix_crop_with_multi_scale')
        elif self.mode == 'eval':
            # ts = kits.random_crop(ts, crop_size)
            # print('eval: random crop')
            ts = kits.crop(ts, crop_size, 'mid')
            print('eval: central crop')

        # flip horizontally when train
        if self.mode == 'train':
            ts = kits.random_flip_left_right(ts, 0.5)

        if VARS['mode'] in ['train', 'eval']:
            return {'X': ts, 'Y': self.raw_inputs['Y']}
        else:
            return {'X': ts, 'Y': self.raw_inputs['Y'],
                    'name': self.raw_inputs['name']}

    def create_threads(self):
        num_thread = self.QUEUE['num_thread']
        # create queue_runners
        VARS['queue_runners'].append(
            tf.train.QueueRunner(self.queue,
                                 [self.enqueue_op] * num_thread))

        # trigger all queue_runners
        self.tf_threads += [qr.create_threads(self.sess, coord=self.coord, start=True)
                            for qr in VARS['queue_runners']]
        return []

    def num_examples(self):
        if self.mode in ['train', 'eval']:
            # one frame a video
            return len(self.example_lst)
        else:
            # eval: num_per_video * 10 * num_videos
            return self.num_per_video * 10 * len(self.example_lst)

if __name__ == '__main__':
    # from inputs import spatial_reader_test, video_kits_test
    from inputs import fast_video_reader_test
    tf.test.main()
