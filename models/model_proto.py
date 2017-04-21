from __future__ import division
import h5py
import numpy as np
import tensorflow as tf

from config.config_agent import FLAGS, VARS
from kits import pp


class Model_proto(object):

    def __init__(self):
        self.global_step = VARS['global_step']
        self.sess = VARS['sess']
        self.num_class = FLAGS['input']['num_class']
        self.OPT = FLAGS['optimizer']
        self.decay_factor = FLAGS['decay_factor']
        self.model_config = FLAGS['model_config']
        self.init_weights_path = self.model_config["init_weights"] \
            if 'init_weights' in self.model_config else None

    def infer(self, inputs):
        raise ValueError('your model need own inference method')

    def get_vars_to_restore(self):
        pp('use vanilla vars_to_restore')
        vars_to_restore = {var.name: var for var in tf.trainable_variables()}
        return vars_to_restore

    def model_init(self):
        # init model weights from pretrained model
        if VARS['if_restart'] is True and self.init_weights_path is not None:

            vars_map = self.get_vars_to_restore()
            except_vars = set(['...' + '/'.join(var.name.split('/')[-3:])
                               for var in tf.global_variables()
                               if var not in vars_map.values()])
            pp('> init except vars\n', except_vars, scope='fine_tune')

            if self.init_weights_path.endswith('.hdf5'):
                self.model_init_from_hdf5(vars_map)
            elif self.init_weights_path.endswith('.npy'):
                self.model_init_from_npy(vars_map)
            else:
                self.model_init_from_ckpt(vars_map)
        else:
            pp('no init from pretrained model')

    def model_init_from_npy(self, vars_map):
        """
        vars_map: {'conv2d/weights': var(weights)}
        """
        pp('restore vars from npy file: %s...' %
           self.init_weights_path)

        w_npy = np.load(self.init_weights_path).item()
        for var_name, var in vars_map.iteritems():
            n_lst0 = '/'.join(var_name.split('/')[:-1])
            n_lst1 = var_name.split('/')[-1]
            self.sess.run(var.assign(w_npy[n_lst0][n_lst1]))
            pp('init > %s' % var_name)
        pp('done')

    def model_init_from_ckpt(self, vars_map):
        """
        vars_map: {'conv2d/weights': var(weights)}
        """
        restorer = tf.train.Saver(vars_map)
        # Restore variables from disk.
        pp('restore vars from ckpt: %s...' % self.init_weights_path)
        restorer.restore(self.sess, self.init_weights_path)
        pp('init > ', [name for name in vars_map])
        pp('done')

    def model_init_from_hdf5(self, vars_map):
        """
        vars_map: {'conv2d/weights': var(weights)}
        """
        pp('restore vars from hdf5 file: %s...' %
           self.init_weights_path)
        with h5py.File(self.init_weights_path, 'r') as f:
            for var_name, var in vars_map.iteritems():
                npy_var = np.array(f[var_name])
                # assign hdf5_weights to tf_vars
                self.sess.run(var.assign(npy_var))
                pp('init > %s' % var_name)
        pp('done')

    def loss(self, logits, labels, scope=None):
        """Add L2Loss to all the trainable variables.

        Add summary for "Loss" and "Loss/avg".
        Args:
          logits: Logits from inference().
          labels: Labels from distorted_inputs or inputs(). 1-D tensor
                  of shape [batch_size]

        Returns:
          Loss tensor of type float.
        """
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        # following function will softmax internally
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss,
        #   plus all of the weight
        # decay terms (L2 loss).
        total_losses = tf.losses.get_losses(
            scope=scope) + tf.losses.get_regularization_losses()
        pp('total losses lst: <%s>...(%d)' %
           (total_losses[0].name, len(total_losses)))
        return tf.add_n(total_losses, name='total_loss')

    def get_lr(self):
        # for train only
        decay_steps = FLAGS['num_steps_per_decay']
        initial_learning_rate = FLAGS['initial_learning_rate']

        if isinstance(decay_steps, (int, long)):
            # Decay the learning rate exponentially based on the number of
            # steps.
            lr = tf.train.exponential_decay(initial_learning_rate,
                                            self.global_step,
                                            decay_steps,
                                            self.decay_factor,
                                            staircase=True)
        elif isinstance(decay_steps, (tuple, list)):
            lr = tf.case([(tf.less(self.global_step, step),
                           lambda i=i: initial_learning_rate * tf.pow(0.1, i))
                          for _, step in enumerate(decay_steps)],
                         default=lambda: initial_learning_rate * tf.pow(0.1, len(decay_steps)))
        else:
            raise ValueError('wrong learning_rate format')

        tf.summary.scalar('learning_rate', lr)
        self.learning_rate = lr

        return lr

    def get_opt(self):
        # for train only
        name = self.OPT['name']
        args = self.OPT.get('args', {})

        lr = self.get_lr()

        if not hasattr(tf.train, '%sOptimizer' % name):
            raise ValueError('%s optimizer not support', name)

        optimizer = getattr(tf.train, '%sOptimizer' % name)
        return optimizer(lr, **args)

    def grad(self, opt, total_loss):
        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = self.add_loss_summaries(total_loss)
        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            grads = opt.compute_gradients(total_loss)

        return grads

    def add_loss_summaries(self, loss):
        """Add summaries for losses in model.

        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.

        Args:
            loss (tensor): total loss
        Returns:
            loss_averages_op: op for generating moving averages of losses.
        """
        # Compute the moving average the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(
            self.moving_average_decay,
            name='avg')
        loss_averages_op = loss_averages.apply([loss])

        # Name each loss as '(raw)' and name the moving average version of
        # the loss as the original loss name.
        tf.summary.scalar(loss.op.name + ' (raw)', loss)
        tf.summary.scalar(loss.op.name, loss_averages.average(loss))

        return loss_averages_op

    def eval(self, logits, labels, top):
        return tf.nn.in_top_k(logits, labels, top)
