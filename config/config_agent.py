import os
import commentjson as cjson
import argparse


def load(config_name, mode=None):
    # load from a config.json
    file_name = config_name + '.json'
    file_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(file_dir, file_name)

    if not os.path.exists(file_path):
        raise ValueError('no such config file: %s'
                         % file_path)

    with open(file_path) as f:
        config = cjson.load(f)

    define_link(config)
    if not mode is None:
        unroll_key(mode, config)

    return config


def init_FLAGS(mode=None):
    # default: config name from stdin
    parser = argparse.ArgumentParser(description='which json file'
                                     ' would you config')
    parser.add_argument('config_name', help='config file name')
    parser.add_argument('--clear', dest='if_restart', action='store_const',
                        const=True, default=False,
                        help='if true, then restart training')
    args = parser.parse_args()

    config = load(args.config_name, mode)

    # mode
    FLAGS['mode'] = mode
    # if restart to train
    FLAGS['if_restart'] = args.if_restart

    for k, v in config.iteritems():
        FLAGS[k] = v


def unroll_key(key, body):
    # arm FLAGS with specific mode (train or eval)
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

# global variable cross modules
FLAGS = {}


def main():
    # for test
    # set_mode('train')
    pass

if __name__ == '__main__':
    main()
