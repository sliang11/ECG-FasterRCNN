import torch
import numpy as np


class call_back():
    @staticmethod
    def detection_collate_RPN(batch):  #
        labels = []
        datas = []
        nums = []
        windows = []
        r_peaks = []
        for data, window, label, r_peak, number in batch:
            for i in range(len(label)):
                window[i].append(label[i] + 1)

            labels.append(torch.Tensor(window))
            data = data.reshape(1, -1)
            datas.append(torch.Tensor(data))
            nums.append(number)
            r_peaks.append(r_peak)
        datas = torch.stack(datas)

        return datas, labels, r_peaks, nums
