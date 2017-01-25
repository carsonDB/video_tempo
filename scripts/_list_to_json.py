"""convert list to json (abs_path, label)

This script is used for UCF-101 stored in frames.
"""
from __future__ import division
import csv
import json
import os


Name2id = {}
Data_root = '/home/wsy/dataset/ucf_101/'


def load_classLst(classInd_path):
    with open(classInd_path, 'r') as f:
        for row in csv.reader(f, delimiter=' '):
            # id--
            Name2id[row[1]] = int(row[0]) - 1

    return Name2id


def convert(src_path):
    data_dir = Data_root + 'rgb_data/'
    re_lst = []
    with open(src_path, 'r') as f:
        for row in csv.reader(f, delimiter=' '):
            video_name = '.'.join(row[0].split('.')[:-1])
            video_label = Name2id[video_name.split('/')[0]]
            video_path = data_dir + video_name
            re_lst.append([video_path, video_label])

    return re_lst


def main():
    # source lst root
    lst_root = '/home/wsy/dataset/ucf_101/ucf101_lst/'
    # classInd txt
    classInd_path = lst_root + 'classInd.txt'
    # name -> label
    load_classLst(classInd_path)

    # lst: trainlist01~03, testlist01~03
    train_lst_arr = ['trainlist%02d' % (i+1) for i in range(3)]
    test_lst_arr = ['testlist%02d' % (i+1) for i in range(3)]
    src_arr = train_lst_arr + test_lst_arr

    # convert
    for src_lst in src_arr:
        src_lst_path = lst_root + src_lst + '.txt'
        print('load %s' % src_lst_path)
        dest_dir = '../data/data_lst'
        lst = convert(src_lst_path)
        json.dump(lst, open('%s/%s.json' % (dest_dir, src_lst), 'w'))

if __name__ == '__main__':
    main()
