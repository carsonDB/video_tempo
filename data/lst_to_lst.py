"""
Convert from an official list of UCF-101
to a suitable list for experiments.
"""

import os
import re
import csv
import json
from random import shuffle


# classInd
CLASS_IND_PATH = '/home/user/data/ucf101_lst/classInd.txt'
# other options
MIN_LEN = 0
IF_SHUFFLE = False
# training dataset of split 1


def gen_name_to_id():
    # class_name maps to id
    name_to_id = {}
    with open(CLASS_IND_PATH, 'r') as f:
        classInd_reader = csv.reader(f, delimiter=' ')
        for row in classInd_reader:
            name_to_id[row[1]] = int(row[0])

    return name_to_id


def lst_to_lst(raw_lst_path, dest_lst_path):

    # get dict: class_name to id
    name_to_id = gen_name_to_id()

    # get raw_lst
    raw_lst = []
    with open(raw_lst_path, 'r') as f:
        raw_lst_reader = csv.reader(f, delimiter=' ')
        for row in raw_lst_reader:
            class_name = row[0].split('/')[0]
            label_id = name_to_id[class_name] - 1
            raw_lst.append([row[0], label_id])

    # if shuffle list
    IF_SHUFFLE = False
    if IF_SHUFFLE:
        shuffle(raw_lst)

    # generate dest_lst
    dest_lst = []
    for item in raw_lst:
        # remove suffix
        item[0] = re.sub('\.avi', '', item[0])
        # convert to absolute path
        item_path = os.path.join(DATA_PATH, item[0])
        if (os.path.exists(item_path)
                and len(os.listdir(item_path))/2 >= MIN_LEN):
            dest_lst.append([item_path, item[1]])

    # write to lst
    with open(dest_lst_path, 'w') as f:
        json.dump(dest_lst, f)


# rgb
DATA_PATH = '/home/user/data/ucf101_rgb_img'

raw_train_path = '/home/user/data/ucf101_lst/trainlist01.txt'
raw_test_path = '/home/user/data/ucf101_lst/testlist01.txt'

# list of training dataset of split 1
dest_train_path = '/home/user/data/train_rgb_ucf101_sp1.lst'
dest_test_path = '/home/user/data/test_rgb_ucf101_sp1.lst'

lst_to_lst(raw_train_path, dest_train_path)
lst_to_lst(raw_test_path, dest_test_path)

# flow
DATA_PATH = '/home/user/data/ucf101_flow_img'

raw_train_path = '/home/user/data/ucf101_lst/trainlist01.txt'
raw_test_path = '/home/user/data/ucf101_lst/testlist01.txt'

# list of training dataset of split 1
dest_train_path = '/home/user/data/train_flow_ucf101_sp1.lst'
dest_test_path = '/home/user/data/test_flow_ucf101_sp1.lst'

lst_to_lst(raw_train_path, dest_train_path)
lst_to_lst(raw_test_path, dest_test_path)
