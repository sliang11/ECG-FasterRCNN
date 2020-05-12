from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import torch.nn as nn
from random import shuffle
import scipy.signal as sig
from math import sqrt
import pickle
import random

from data_ import filter_data, random_10s_beat_align, aligned


class my_dataset(Dataset):
    def __init__(self, path=os.getcwd() + '/data/', test=False, eval=False, all_data=None, all_label=None, index_=None):

        self.path = path
        self.train = os.listdir(path)
        self.train = [i.split('_')[0] for i in self.train]
        self.train = list(set(self.train))
        self.train.sort()
        self.test = test
        self.train = self.train
        self.data = []
        self.label = []
        self.eval = eval

        # print('path:{}'.format(self.path))
        max_R = 169
        max_len = 500
        ####################################_
        if index_ == None:
            for index in self.train:
                data = np.load(self.path + '{}_data.npy'.format(index))
                # for lead in range(data.shape[-1]):
                #     data[:, lead] = filter_data(data[:, lead], average=True)
                data = filter_data(data, average=True)
                data = data.reshape(1, -1)
                label = np.load(self.path + '{}_detect_label.npy'.format(index))
                self.data.append(data)

                label = [i for i in label if i[1] - i[0] < 1400 and i[-1] != 4 and i[1] - i[0] != 1175]
                for j in range(len(label)):
                    start = label[j][0]
                    end = label[j][1]
                    R_loc = label[j][2]
                    if end - start > 250:
                        if R_loc - start > 100:
                            start = R_loc - 100
                            if end - R_loc > 200:
                                end = R_loc + 200
                        elif end - R_loc > 200:
                            end = R_loc + 200
                    label[j][0] = start
                    label[j][1] = end
                    label[j][2] = R_loc - start
                self.label.append(label)
            print(max_len)
            print(max_R)
            tmp_data1, tmp_label1 = random_10s_beat_align(self)
            tmp_data1, tmp_label1 = aligned(tmp_data1, tmp_label1, max_R, max_len)
            self.data = tmp_data1.copy()
            self.label = tmp_label1.copy()
            # #
            tmp_data = []
            for i in self.data:
                i = [ii.transpose() for ii in i]
                tmp_data.extend(i)

            self.data = tmp_data
            self.data = [i.reshape(-1) for i in self.data]
            index_normal = [i for i in range(len(self.label)) if self.label[i] == 0]
            from random import shuffle
            shuffle(index_normal)
            index2 = [i for i in range(len(self.label)) if self.label[i] != 0]
            index3 = index_normal[:int(0.15 * len(index_normal))]
            index3.extend(index2)
            # normal_data = np.asarray([self.data[i] for i in index_normal])
            # normal_label = np.asarray([self.label[i] for i in index_normal])
            self.data = [self.data[i] for i in index3]
            self.label = [self.label[i] for i in index3]
            self.data = np.asarray(self.data)
            self.label = np.asarray(self.label)
            from imblearn.combine import SMOTETomek
            smt = SMOTETomek()
            self.data, self.label = smt.fit_resample(self.data, self.label)
            #
            # self.data = np.vstack([self.data, normal_data])
            # self.label = np.hstack([self.label, normal_label])
        else:
            self.data = all_data
            self.label = all_label
        total = len(self.label)

        if index_ == None:
            total = len(self.label)
            total_list = list(range(total))
            shuffle(total_list)
        else:
            total_list = index_
        self.train_index = total_list[:int(0.8 * total)]
        self.test_index = total_list[-int(0.2 * total):]
        self.all_data = self.data
        self.all_label = self.label
        self.index = total_list
        if eval or test:
            self.data = [self.data[i] for i in self.test_index]
            self.label = [self.label[i] for i in self.test_index]
            self.data = [i.reshape(500, 1) for i in self.data]
        else:
            self.data = [self.data[i] for i in self.train_index]
            self.label = [self.label[i] for i in self.train_index]
            self.data = [i.reshape(500, 1) for i in self.data]

    def __getitem__(self, item):
        tmp_label = self.label[item]
        tmp_data = self.data[item]
        return tmp_data, tmp_label

    def __len__(self):
        return len(self.data)
