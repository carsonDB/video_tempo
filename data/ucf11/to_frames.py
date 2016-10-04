from __future__ import division

import os
import cv2
import csv
import numpy as np
import json
import commentjson as cjson

DEST_PATH = '/home/wsy/dataset/frames'
OUTPUT_LST = []

# read from data.json
with open('data.json', 'r') as f:
    FLAGS = cjson.load(f)


def extract_seq_frames(path, label, mode):
    num_group = FLAGS['num_group']
    stride = FLAGS['stride']
    prefix = '.'.join(path.split('/')[-1].split('.')[:-1])

    # return groups of seq_frames from one video
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("can't open: %s" % path)

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # for the safe length ...
    frame_count -= 2
    if frame_count < num_group*stride:
        return None

    for i in range(stride):
        for j in range(num_group):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i + j*stride)
            ret, frm = cap.read()
            if not ret:
                raise ValueError("""
                  No.%d frame not available,
                  at file: %s
                  whose length is: %d
                  """ % (i + j*stride, path, frame_count))
            frm = cv2.resize(frm,
                             tuple(reversed(FLAGS['resolution'])))

            frame_name = '%s_f%d.jpg' % (prefix, i + j*stride)
            dest_frame_path = os.path.join(DEST_PATH + '_' + mode, frame_name)

            if os.path.isfile(dest_frame_path):
                raise ValueError('file %s has already existed',
                                 dest_frame_path)
            cv2.imwrite(dest_frame_path, frm)
            OUTPUT_LST.append([dest_frame_path, label])


def read_lst(path):
    with open(path, 'r') as f:
        return json.load(f)

# read list
train_lst_path = '/home/wsy/dataset/train_lst'
test_lst_path = '/home/wsy/dataset/test_lst'
train_lst = read_lst(train_lst_path)
test_lst = read_lst(test_lst_path)

# gen_frames from list
for i, item in enumerate(train_lst):
    print(i)
    frms_lst = extract_seq_frames(item[0], item[1], 'train')

# write into file
with open(DEST_PATH + '_train.lst', 'w') as f:
    writer = csv.writer(f)
    for row in OUTPUT_LST:
        writer.writerow(row)
