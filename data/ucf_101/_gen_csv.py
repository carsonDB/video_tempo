import os
import sys
import csv
import random


UCF101_DIR = '/home/wsy/dataset/ucf_101/'
UCF101_LST_DIR = UCF101_DIR + 'ucf101_lst/'
UCF101_DATA_DIR = UCF101_DIR + 'rgb_data/'

TRAIN_S1_PATH = UCF101_LST_DIR + 'trainlist01.txt'
TEST_S1_PATH = UCF101_LST_DIR + 'testlist01.txt'

Name2id = {}


def load_classLst(classInd_path):
    with open(classInd_path, 'r') as f:
        for row in csv.reader(f, delimiter=' '):
            # id--
            Name2id[row[1]] = str(int(row[0]) - 1)

    return Name2id


def read_csv(path):
    """read official ucf101_lst
    format: (video_path, label<1..>)
    """
    lst = []
    with open(path, 'r') as f:
        for row in csv.reader(f, delimiter=' '):
            # in case testlist without labels
            classLabel = Name2id[row[0].split('/')[0]]
            if len(row) == 2:
                assert(row[1], classLabel)
            lst.append([row[0], classLabel])

    return lst


def write_csv(lst, path):
    """write ucf101_lst read by Xiong's caffe
    format: (video_path, length, label<0..>)
    """
    with open(path, 'w') as f:
        fwriter = csv.writer(f, delimiter=' ')
        for row in lst:
            fwriter.writerow(row)


def gen_a_lst(src_path, dst_path, data_dir):
    src_lst = read_csv(src_path)
    dst_lst = []
    # (./video_path, label) -> (/../video_path, length, label)
    for video_path, label in src_lst:
        frames_dir = '.'.join(video_path.split('.')[:-1])
        frames_dir = data_dir + frames_dir
        # only for rgb
        length = len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        length = str(length)
        print(frames_dir, length, label)
        dst_lst.append([frames_dir, length, label])

    # shuffle
    random.shuffle(dst_lst)
    write_csv(dst_lst, dst_path)


if __name__ == '__main__':
    load_classLst(UCF101_LST_DIR + 'classInd.txt')

    # for split_01
    train_split1 = 'trainlist01.txt'
    gen_a_lst(UCF101_LST_DIR + train_split1,
              './train_rgb_split1.txt', UCF101_DATA_DIR)

    train_split1 = 'testlist01.txt'
    gen_a_lst(UCF101_LST_DIR + train_split1,
              './val_rgb_split1.txt', UCF101_DATA_DIR)
