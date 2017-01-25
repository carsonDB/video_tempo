import tensorflow as tf

# import local files
from video_tempo.config.config_agent import FLAGS
from video_tempo.models.model_proto import Model_proto
from video_tempo.models.googlenet import GoogleNet
from video_tempo.models.cnn import Model as CNN


class Model(Model_proto):
    """GoogleNet with a bunch of probes

    Probes are pulgged from inputs to last conv layer.

    """
    def __init__(self):
        super(Model, self).__init__()
        self.plugged_layer_name = FLAGS['plugged_layer']

    def infer(self, inputs):
        """Summary

        Args:
            inputs (tensor): [batch, depth, 224, 224, 3]

        Returns:
            tensor: output from a probe
        """
        # raw_shape = inputs.get_shape().as_list()
        # # reshape from [b, d, h, w, c] -> [b+d, h, w, c]
        # re_inputs = tf.reshape(inputs, [-1] + raw_shape[-3:])
        # net = GoogleNet({'data': re_inputs})
        # # specific layer plugged in
        # plugged_layer = tf.stop_gradient(net.layers[self.plugged_layer_name])

        c3d = CNN()
        # reshape from [b+d, h, w, c] -> [b, d, h, w, c]
        # out_shape = plugged_layer.get_shape().as_list()
        # l_out = tf.reshape(plugged_layer, raw_shape[:2] + out_shape[-3:])
        l_out = c3d.infer(inputs)

        # # followed by a linear classifier
        # # reshape from [b+d, h, w, c] -> [b, d+h+w+c]
        # l_out = tf.reshape(plugged_layer, raw_shape[:1] + [-1])

        # probe_name = 'probe_%s' % self.plugged_layer_name
        # l_out = affine_transform(l_out, self.num_class,
        #                          probe_name)
        return l_out
