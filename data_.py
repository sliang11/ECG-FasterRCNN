import torch
import numpy as np
import pickle
import os
import random
import scipy.signal as sig
from sklearn.preprocessing import scale
from imblearn.combine import SMOTETomek


def detection_collate(batch):
    label = []
    data = []
    # su = 0
    for sin_data, sin_label in batch:
        if sin_label != 4 and sin_label != -1:
            label.append(sin_label)
            data.append(sin_data)


    every_len = [len(i) for i in data]
    max_len = max(every_len)
    data = [np.pad(i, ((0, max_len - len(i)), (0, 0)), 'constant', constant_values=0).transpose()
            for i
            in data]
    data = [torch.Tensor(i) for i in data]
    data = torch.stack(data)
    every_len = [int(i / 64) for i in every_len]
    return data, torch.LongTensor(label), torch.Tensor(every_len), max_len


def proprecess(self):
    a = 0
    count = 0
    for index in self.train:
        data = np.load(self.path + '{}_data.npy'.format(index))
        for lead in range(data.shape[-1]):
            data[:, lead] = filter_data(data[:, lead], average=True)
        data = data[:, 0]
        data = data.reshape(1, -1)
        label = np.load(self.path + '{}_detect_label.npy'.format(index))
        self.data.append(data)



        label = [i for i in label]
        count += len(label)
        for j in range(len(label)):
            start = label[j][0]
            end = label[j][1]
            R_loc = label[j][2]
            if label[j][-1] == 4:  # 2019.11.10
                label[j][-1] = -1
            if end - start > 250:
                if R_loc - start > 100:
                    start = R_loc - 100
                    if end - R_loc > 200:
                        end = R_loc + 200
                elif end - R_loc > 200:
                    end = R_loc + 200
            label[j][0] = start
            label[j][1] = end
            # label[j][2] = R_loc - start
            label[j][2] = R_loc

            if end - start > a:
                a = end - start
        self.label.append(label)
    print(count)
    # print("max_len:{}".format(a))


def filter_data(data, average=False, norm=False):
    b, a = sig.butter(3, 0.25, 'lowpass')
    d, c = sig.butter(3, 0.0015, 'highpass')
    tmp = data
    fil_data = sig.filtfilt(b, a, tmp)
    fil_data = sig.filtfilt(d, c, fil_data)

    if average == True:
        data = fil_data - np.average(fil_data)
    else:
        data = scale(data)
    return data




def aligned(data, label, R_loc, max_len):
    for index1 in range(len(data)):
        for index2 in range(len(data[index1])):
            this_len = R_loc - label[index1][index2][0]
            remian_len = max_len - this_len - data[index1][index2].shape[1]
            data[index1][index2] = np.pad(data[index1][index2], ((0, 0), (this_len, remian_len)), mode='constant')

    # label = [[j[-1] for j in i] for i in label]
    label = [j[-1] for i in label for j in i]

    return data, label


def random_10s_beat_align(self, level=1):
    count = 0

    length = 3600
    tmp_data = []
    tmp_label = []
    for i in range(len(self.data)):
        this_data = self.data[i]
        this_label = self.label[i]
        this_len = this_data.shape[-1]
        each_seg_length = length
        segment = int(this_len / each_seg_length)
        for j in range(segment):
            this_seg_label = [
                [this_label[k][0] - j * each_seg_length, this_label[k][1] - j * each_seg_length, this_label[k][-2],
                 this_label[k][-1]]
                for k in range(len(this_label))
                if
                this_label[k][0] >= j * each_seg_length and this_label[k][1] <= (
                            j + 1) * each_seg_length + 400]

            this_seg_data = [this_data[0][k[0] + j * each_seg_length:k[1] + j * each_seg_length].reshape(1, -1) for k in
                             this_seg_label]

            this_seg_label = [[i[-2], i[-1]] for i in this_seg_label]
            if len(this_seg_label) > 0:
                tmp_data.append(this_seg_data)
                tmp_label.append(this_seg_label)
            else:
                count += 1

    return tmp_data, tmp_label
