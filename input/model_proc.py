from importlib import import_module
import tensorflow as tf

# import local file
from config import config_agent


def feature_extract(ts, opFLAGS, sess):
    # preprocess through model
    config_name = opFLAGS['config_name']
    ckpt_path = opFLAGS['ckpt_path']
    in_port = opFLAGS['in_port']

    # determin which model to preprocess
    preFLAGS = config_agent.load(config_name, 'train')
    model_type = preFLAGS['type']
    nn = import_module('model.' + model_type)

    # determin which layer to output
    out_layer_id = None
    if ('out_port' in preFLAGS
            and in_port in preFLAGS['out_port']):
        out_layer_id = preFLAGS['out_port'][in_port]

    # build pretrained model
    logits = nn.inference(ts, preFLAGS, out_layer_id)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        preFLAGS['moving_average_decay'])
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # restore pre-trained model variables (trainable == False)
    with sess:
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found')

    return logits
