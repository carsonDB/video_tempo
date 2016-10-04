"""
This script is only for test.
Not the part of the project.
"""
import tensorflow as tf
from tensorflow.python.ops.rnn import raw_rnn

from model import kits

max_time = 30
batch_size = 8
input_depth = 1024
num_units = 30

inputs = tf.placeholder(shape=(max_time, batch_size, input_depth),
                        dtype=tf.float32)
labels = tf.placeholder(shape=(batch_size,), dtype=tf.int32)
sequence_length = tf.placeholder(shape=(batch_size, max_time), dtype=tf.int32)
inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
inputs_ta = inputs_ta.unpack(inputs)


def loop_fn(time, cell_output, loop_state):
    emit_output = cell_output  # == None for time == 0
    elements_finished = (time >= sequence_length)
    finished = tf.reduce_all(elements_finished)
    next_input = tf.cond(
        finished,
        lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
        lambda: inputs_ta.read(time))
    next_loop_state = None
    return (elements_finished, next_input, emit_output, next_loop_state)

cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs_ta, final_state, _ = raw_rnn(cell, loop_fn, initial_state)
outputs = outputs_ta.pack()


loss_op = kits.loss(outputs, labels)

import pdb; pdb.set_trace()  # breakpoint 49561a3a //
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

sess.run(loss_op)