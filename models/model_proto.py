from __future__ import division
import tensorflow as tf

from config.config_agent import FLAGS, VARS


class Model_proto(object):

    def __init__(self):
        self.global_step = VARS['global_step']
        self.num_class = FLAGS['input']['num_class']
        self.moving_average_decay = FLAGS['moving_average_decay']

    def infer(self):
        raise ValueError('your model need own inference method')

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
        return tf.add_n(tf.get_collection('losses', scope=scope),
                        name='total_loss')

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
