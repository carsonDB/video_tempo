from __future__ import division
import sys
from importlib import import_module
import traceback
import tensorflow as tf

from config.config_agent import FLAGS, VARS


class Solver(object):

    def __init__(self):
        self.batch_size = FLAGS['batch_size']
        self.gpus = FLAGS['gpus']
        self.OPT = FLAGS['optimizer']
        self.moving_average_decay = FLAGS['moving_average_decay']
        self.decay_factor = FLAGS['decay_factor']
        self.if_restart = VARS['if_restart']
        self.ckpt_dir = FLAGS['ckpt_dir']
        self.model_name = FLAGS['model']
        self.reader_name = FLAGS['input']['reader']
        self.dest_dir = FLAGS['dest_dir']

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
                # self.graph.finalize()
                self.reader.launch()
                self.launch_graph()
                print('%s process closed normally\n' % VARS['mode'])
            except:
                traceback.print_exc(file=sys.stdout)
                print('%s process closed with error\n' % VARS['mode'])
            finally:
                self.reader.close()

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

        # determine reader and model
        # and instantiate a read and a model
        model_module = import_module('models.' + self.model_name)
        read_module = import_module('inputs.' + self.reader_name)
        self.model = model_module.Model()
        self.reader = read_module.Reader()

    def get_opt(self):
        name = self.OPT['name']
        args = self.OPT['args'] if 'args' in self.OPT else {}

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(self.initial_learning_rate,
                                        self.global_step,
                                        self.decay_steps,
                                        self.decay_factor,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)
        self.learning_rate = lr

        if not hasattr(tf.train, '%sOptimizer' % name):
            raise ValueError('%s optimizer not support', name)

        optimizer = getattr(tf.train, '%sOptimizer' % name)
        return optimizer(lr, **args)

    def init_sess(self):
        # initialize variables
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)

    def init_graph(self):
        # Build the summary operation based on the TF collection of Summaries.
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables())
        # init session
        self.init_sess()
        # restore variables if any
        if self.if_restart is False:
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        self.summary_writer = tf.summary.FileWriter(self.dest_dir,
                                                    self.sess.graph)
