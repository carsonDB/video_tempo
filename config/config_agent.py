from __future__ import division
import os
import re
import cson
import argparse


def load(config_name, mode=None):
    # load from a config.json
    file_name = config_name + '.cson'
    file_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(file_dir, file_name)

    if not os.path.exists(file_path):
        raise ValueError('no such config file: %s'
                         % file_path)

    with open(file_path) as f:
        config = cson.load(f)

    define_link(config)

    return config


def init_FLAGS(mode=None):
    # default: config name from stdin
    parser = argparse.ArgumentParser(description='which json file'
                                     ' would you config')
    parser.add_argument('config_name', help='config file name')
    parser.add_argument('--clear', dest='if_restart', action='store_const',
                        const=True, default=False,
                        help='if true, then restart training')
    parser.add_argument('--test', dest='if_test', action='store_const',
                        const=True, default=False,
                        help='if true, test. If not, validate')
    args = parser.parse_args()

    # # mode
    # VARS['mode'] = mode

    # if restart to train
    VARS['if_restart'] = args.if_restart
    VARS['if_test'] = args.if_test

    config = load(args.config_name, mode)
    # specific mode variables localize
    if mode is not None:
        unroll_key(mode, config)
    # # sub-mode of eval
    # if mode == 'eval':
    #     if args.if_test:
    #         unroll_key('test', config)
    #     else:
    #         unroll_key('valid', config)

    for k, v in config.iteritems():
        FLAGS[k] = v


def unroll_key(key, body):
    # arm FLAGS with specific mode (train or eval)
    key = '@' + key
    if not key in body:
        raise ValueError('no such mode: %s' % key)

    sub_body = body[key]
    for k, v in sub_body.iteritems():
        body[k] = v


def define_link(config):
    # pre-defined macro link
    if not '__define__' in config:
        return
    defined_dict = config['__define__']

    # substitute macro with body
    def unroll(obj):
        # check if exist reference
        while (isinstance(obj, basestring) and
                obj in defined_dict):
            obj = defined_dict[obj]

        # list or dict (containing reference)
        if (isinstance(obj, (list, tuple)) and
                not isinstance(obj, basestring)):
            for i, v in enumerate(obj):
                obj[i] = unroll(v)
        elif isinstance(obj, dict):
            for k, v in obj.iteritems():
                obj[k] = unroll(v)

        return obj

    for k, v in config.iteritems():
        config[k] = unroll(v)

# global variable(configure) cross modules
FLAGS = {}
# global variable(build-time vars) cross modeuls
VARS = {
    'threads': [],
    'queues': [],
    'output': {},
    'queue_runners': [],
    'summaries': {}
}


def main():
    # only for code test
    init_FLAGS('eval')
    print(FLAGS['num_per_video'])
    print(FLAGS['__define__']['DROPOUT'])

if __name__ == '__main__':
    main()
