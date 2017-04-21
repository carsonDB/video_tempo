from __future__ import print_function
from __future__ import division
import tensorflow as tf

from config.config_agent import FLAGS, VARS


class Input_proto(object):
    """prototype of Reader:
        * build start nodes and preprocess nodes
        * build read queue
        * manage read process
    """

    def __init__(self):
        self.QUEUE = FLAGS['input_queue']
        self.mode = VARS['mode']
        self.INPUT = FLAGS['input']
        self.queue_type = self.QUEUE['type']
        self.capacity = self.QUEUE['capacity']
        self.input_batch_size = VARS['input_batch_size']
        self.num_thread = self.QUEUE['num_thread']
        self.num_class = self.INPUT['num_class']
        self.example_size = self.INPUT['example_size']
        self.sess = VARS['sess']
        self.coord = VARS['coord']
        self.if_test = VARS['if_test']

        self.py_threads = []
        self.tf_threads = []

    def read(self):
        # build start nodes (e.g. placeholder) and preprocesss
        inputs = self.get_data()
        # async (e.g. through a queue)
        # inputs -> inputs_batch
        inputs_batch = self.async(inputs)

        # create reading threads
        self.threads = self.create_threads()
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        return inputs_batch

    def async(self, inputs):
        # calculate shape and type of inputs
        inputs_keys = [k for k in inputs]
        inputs_shape = [inputs[k].get_shape() for k in inputs]
        inputs_type = [inputs[k].dtype for k in inputs]

        # build a queue
        if self.queue_type == 'shuffle':
            min_remain = self.QUEUE['min_remain']
            q = tf.RandomShuffleQueue(self.capacity, min_remain,
                                      inputs_type,
                                      shapes=inputs_shape,
                                      names=inputs_keys)
        else:
            q = tf.FIFOQueue(self.capacity, inputs_type,
                             shapes=inputs_shape,
                             names=inputs_keys)

        self.queue = q
        # queue add to tensorboard
        self.queue_state = q.size() / self.capacity  # multi-gpus

        # enqueue an example
        self.enqueue_op = q.enqueue(inputs)
        # dequeue a batch of examples
        inputs_batch = q.dequeue_many(self.input_batch_size)

        return inputs_batch

    def launch(self):
        # start tensorflow queue_runners
        self.tf_threads = tf.train.start_queue_runners(
            sess=self.sess, coord=self.coord)
        # start customed runners
        for t in self.py_threads:
            t.start()

    def close(self):
        # disable equeue op, in case of readers blocking
        self.coord.request_stop()
        self.sess.run(self.close_op)
        self.coord.join(self.py_threads + self.tf_threads)
        self.sess.close()
