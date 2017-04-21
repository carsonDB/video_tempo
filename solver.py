from __future__ import division
from __future__ import print_function
import sys
from importlib import import_module
import traceback
import tensorflow as tf
from tensorflow.python.client import timeline

from config.config_agent import FLAGS, VARS
from kits import pp


class Solver(object):

    def __init__(self):
        self.run_mode = FLAGS['run_mode']
        # input_batch_size
        self.batch_size = FLAGS['batch_size']
        self.iter_size = FLAGS['iter_size'] if VARS['mode'] == 'train' else 1
        assert self.batch_size % self.iter_size == 0, \
            'batch_size must can be divided by iter_size'
        self.input_batch_size = VARS['input_batch_size'] = \
            self.batch_size // self.iter_size

        self.gpus = FLAGS['gpus']
        self.if_restart = VARS['if_restart']
        self.ckpt_dir = FLAGS['ckpt_dir']
        self.model_name = FLAGS['model']
        self.reader_name = FLAGS['input']['reader']
        self.dest_dir = FLAGS['dest_dir']

        self.summaries = None
        self.run_options = None
        self.run_metadata = None
        self.profile_log = FLAGS.get('profile_log')

    def build_graph(self):
        raise ValueError('no build_graph method')

    def launch_graph(self):
        raise ValueError('no launch_graph method')

    def start(self):
        # dest_dir
        if not tf.gfile.Exists(self.dest_dir):
            # only start from scratch
            tf.gfile.MakeDirs(self.dest_dir)
            self.if_restart = True
        elif self.if_restart:
            # clear old train_dir and restart
            tf.gfile.DeleteRecursively(self.dest_dir)
            tf.gfile.MakeDirs(self.dest_dir)

        with tf.Graph().as_default() as self.graph, tf.device('/cpu:0'):
            try:
                self.init_env()
                self.build_graph()
                self.init_graph()
                self.graph.finalize()  # freeze graph
                self.reader.launch()
                self.launch_graph()
                print('%s process closed normally\n' % VARS['mode'])
            except:
                traceback.print_exc(file=sys.stdout)
                print('%s process closed with error\n' % VARS['mode'])
            finally:
                self.end()

    def init_env(self):
        # create a session and a coordinator
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        VARS['sess'] = self.sess = tf.Session(config=config)
        VARS['coord'] = self.coord = tf.train.Coordinator()
        self.global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        VARS['global_step'] = self.global_step

        # performance profile
        if self.run_mode == 'profile':
            self.run_options = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()

        # determine reader and model
        # and instantiate a read and a model
        model_module = import_module('models.' + self.model_name)
        read_module = import_module('inputs.' + self.reader_name)
        self.model = model_module.Model()
        self.reader = read_module.Reader()

    def run_sess(self, fetches, feed_dict=None):
        return self.sess.run(fetches, feed_dict,
                             options=self.run_options,
                             run_metadata=self.run_metadata)

    def init_sess(self):
        # initialize variables
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)
        print('initialized all variables')

    def init_graph(self):
        """only used for training"""
        assert VARS['mode'] == 'train'

        # Build the summary operation based on the TF collection of Summaries.
        self.summary_op = (tf.summary.merge(self.summaries + VARS['summaries'].values())
                           if hasattr(self, 'summaries') else tf.summary.merge_all())
        self.saver = tf.train.Saver(tf.global_variables())
        # init session
        self.init_sess()
        # restore variables if any
        if self.if_restart is False:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('solver: restore from %s' % ckpt.model_checkpoint_path)

        elif hasattr(self.model, 'model_init'):
            self.model.model_init()

        self.summary_writer = tf.summary.FileWriter(self.dest_dir,
                                                    self.sess.graph)

    def end(self):
        self.reader.close()

        if self.run_mode == 'profile':
            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(self.run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            pp('profile_log write to %s...' % self.profile_log)
            with open(self.profile_log, 'w') as f:
                f.write(ctf)
