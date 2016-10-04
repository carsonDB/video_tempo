import os
import argparse
from random import shuffle
import commentjson as cjson


# items in subdir of root are classes
def get_lst(root, dest_lst):
    lst = []
    classes = os.listdir(root)
    classes = [c for c in classes if not c.endswith('.txt')]

    for i, label in enumerate(classes):
        for subroot, dirs, files in os.walk(os.path.join(root, label)):
            for file in files:
                if file.endswith(('.xgtf', '.txt', '.mpg')):
                    continue
                lst.append((os.path.join(subroot, file), i))

    return lst


def main():
    parser = argparse.ArgumentParser(description='which json file'
                                     ' would you config')
    parser.add_argument('config_path', help='data config file path')
    config_path = parser.parse_args().config_path
    # read from data.json
    with open(config_path, 'r') as f:
        args = cjson.load(f)

    # convert to real path
    root, dest_path = args['data_dir'], args['lst_path']
    root = os.path.expanduser(root)
    dest_path = os.path.expanduser(dest_path)

    # get whole data
    lst = get_lst(root, dest_path)
    if args['shuffle']:
        shuffle(lst)

    # split
    split_lst = args['splits']
    new_lst = {}
    sum_ratio = 0
    for k, v in split_lst.iteritems():
        sum_ratio += v
        new_lst[os.path.expanduser(k)] = int(round(v * len(lst)))
    if sum_ratio > 1:
        raise ValueError('sum of your splits > 1')

    # write into split files
    offset = 0
    for k, v in new_lst.iteritems():
        with open(k, 'w') as f:
            v += offset
            cjson.dump(lst[offset:v-1], f)
            offset = v


if __name__ == '__main__':
    main()
