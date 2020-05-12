import os
import wfdb
from collections import Counter
import numpy as np
import math
import pickle

import matplotlib.pyplot as plt

#
path = 'D:/data/mitbih/'
N = ['L', 'R', 'e', 'j', 'N']
S = ['a', 'A', 'S', 'J']
V = ['E', 'V']
F = ['F']
Q = ['/', 'Q', 'f']
beat_ann = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
none_beat = ['[', ']', '!', 'x', '(', ')', 'p', 't', 'u', '`', ',', '^', '|', '~', '+', 's', 'T', '*', 'D', '=', '"',
             '@']


def return_label(label):
    if label in N:
        return 0
    if label in S:
        return 1
    if label in V:
        return 2
    if label in F:
        return 3
    if label in Q:
        return 4


# all_file = os.listdir(path)
# all_file.sort()
# for this_file in all_file:
#     if '114' in this_file:
#         tmp = open(path + this_file).readlines()
#         new_label = []
#         new_time = []
#         data = tmp[1].strip().split()[1:]
#         data = [float(i) for i in data]
#         data = np.asarray(data).reshape(-1, 1)
#     else:
#         tmp = open(path + this_file).readlines()
#         new_label = []
#         new_time = []
#         data = tmp[0].strip().split()[1:]
#         data = [float(i) for i in data]
#         data = np.asarray(data).reshape(-1, 1)
# label = tmp[3].strip().split()[1:]
# label = [str(i) for i in label]
# time = tmp[2].strip().split()[1:]
# time = [int(i) for i in time]

import wfdb
import os

path = os.path.join(os.getcwd(), 'data_2')

all_files = os.listdir(path)
all_files = [i.split('.')[0] for i in all_files if 'hea' in i]
from collections import Counter

for this_file in all_files:
    file_path = os.path.join(path, this_file)
    if "102" in file_path or "104" in file_path or "107" in file_path or "217" in file_path:
        continue
    ann = wfdb.rdann(file_path, 'atr')
    if '114' not in file_path:
        data = wfdb.rdrecord(file_path).p_signal[:, 0]
    else:
        data = wfdb.rdrecord(file_path).p_signal[:, 0]
    time = ann.sample
    label = ann.symbol
    new_label = []
    new_time = []
    detect_label = []
    for index, i in enumerate(label):
        if i in beat_ann:
            t = return_label(i)
            new_label.append(t)
            new_time.append(time[index])

    for i in range(len(new_time)):  #
        if i != len(new_time) - 1:
            tmp = new_time[i] * 0.4 + new_time[i + 1] * 0.6
        else:
            tmp = new_time[i] + 200 if new_time[i] + 200 < 650000 else 650000

        new_time[i] = [new_time[i], round(tmp)]
        # new_time[-1] = [new_time, len(data)]
    new_time = [i for i in new_time if isinstance(i, list)]

    for index in range(len(new_label)):
        if index == 0:
            start = new_time[0][0] - 100 if new_time[0][0] > 100 else 0
        else:
            start = new_time[index - 1][-1]
        if index == len(new_label) - 1:
            end = new_time[index][0] + 200 if new_time[index][0] + 200 < 650000 else 650000
        else:
            end = new_time[index][-1]
        this_label = new_label[index]
        R = new_time[index][0]
        # y[start:end] = this_label
        detect_label.append([start, end, R, this_label])

    new_time = np.asarray(new_time)

    detect_label = np.asarray(detect_label)
    detect_label = detect_label.astype(np.int)
    np.save(os.path.join(os.getcwd() + '/data', '{}_time'.format(this_file)), new_time)
    np.save(os.path.join(os.getcwd() + '/data', '{}_data'.format(this_file)), data)
    np.save(os.path.join(os.getcwd() + '/data', '{}_detect_label'.format(this_file)), detect_label)
