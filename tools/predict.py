"""feature_extract script
Any tensor storing in VARS['output'] will be available for output.
Running mode is the same with eval script (either valid or test).

All will be float32 by default

Output will be hdf5 files under ./data/.

"""
from __future__ import division
import os
import math
import h5py
import tensorflow as tf

from config import config_agent
from config.config_agent import FLAGS, VARS
from solver import Solver


class Output_solver(Solver):

    def __init__(self):
        self.req_output_lst = FLAGS['output']
        self.avail_output_dict = VARS['output']
        self.output_path = FLAGS['output_path']
        self.num_examples = FLAGS['num_examples']
        super(Output_solver, self).__init__()

    def init_graph(self):
        self.init_sess()
        self.saver = tf.train.Saver(tf.global_variables())
        # restore variables if any
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def build_graph(self):
        with tf.device('/gpu:%d' % self.gpus[-1]):
            # Build a Graph that computes the logits predictions.
            inputs = self.reader.read()
            # @temp
            # VARS['output']['raw_data'] = inputs['raw_data']
            VARS['output']['name'] = inputs['name']
            VARS['output']['label'] = inputs['Y']
            # inference model.
            logits = self.model.infer(inputs['X'])
            # scale logits
            VARS['output']['scaled_logits'] = tf.nn.softmax(logits)

    def launch_graph(self):
        if self.num_examples % self.batch_size != 0:
            print("Warning: num_examples can't be divided by"
                  "batch_size with no remainder")
        # filter tensors to be predicted
        predicted_name_lst = []
        predicted_ts_lst = []
        for k in self.req_output_lst:
            if k not in self.avail_output_dict:
                raise ValueError('%s is not in available output list' % k)
            predicted_name_lst.append(k)
            predicted_ts_lst.append(self.avail_output_dict[k])
        # calculate iterations
        num_iter = int(math.ceil(self.num_examples / self.batch_size))
        total_sample_count = num_iter * self.batch_size
        step = 0
        print('Total steps: %d' % num_iter)

        # write into a hdf5 file
        if tf.gfile.Exists(self.output_path):
            tf.gfile.Remove(self.output_path)
        tf.gfile.MakeDirs(os.path.dirname(self.output_path))

        with h5py.File(self.output_path, 'w') as f:
            # create datasets
            dset_lst = [f.create_dataset(k,
                                         [total_sample_count]
                                         + self.avail_output_dict[k].get_shape().as_list()[1:])
                        for k in predicted_name_lst]
            while step < num_iter and not self.coord.should_stop():
                predicted_np_lst = self.sess.run(predicted_ts_lst)
                for i in range(len(predicted_name_lst)):
                    dset = dset_lst[i]
                    dset[step * self.batch_size:(step + 1) * self.batch_size] = \
                        predicted_np_lst[i]

                step += 1

        print('predict succeed!')
        print(predicted_name_lst)


def main(argv=None):
    # unroll arguments of prediction
    config_agent.init_FLAGS('eval')
    Output_solver().start()

if __name__ == '__main__':
    tf.app.run()
