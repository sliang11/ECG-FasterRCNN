from preprocessing import prerprocessor
from collections import Counter
import numpy as np



class corrector(prerprocessor):

    def correct(self):
        bias = 64
        max_len = 0
        all_len = []
        remove = []
        before = 100
        after = 200
        for i in range(len(self.time)):
            sin_time = self.time[i]
            sin_label = self.label[i]
            sin_r_peak = self.r_peak[i]
            ################
            a = sin_time[0][0]
            b = sin_time[0][1]
            peak = sin_r_peak[0]
            if peak - a > before:
                a = peak - before
            if b - peak > after:
                b = peak + after
            start = a
            sin_time[0] = [0, b - a]
            #########
            for j in range(1, len(sin_time) - 1):
                a = sin_time[j][0]
                b = sin_time[j][1]
                peak = sin_r_peak[j]
                if peak - a > before:
                    a = peak - before
                if b - peak > after:
                    b = peak + after
                a = a - start
                b = b - start
                sin_time[j] = [a, b]

            a = sin_time[-1][0]
            b = sin_time[-1][1]
            peak = sin_r_peak[-1]
            if peak - a > before:
                a = peak - before
            if b - peak > after:
                b = peak + after
            end = b
            a = a - start
            b = b - start
            sin_time[-1] = [a, b]
            for j in range(len(sin_r_peak)):
                sin_r_peak[j] = sin_r_peak[j] - start
            ###################
            # max_len = max(end - start, max_len)
            all_len.append(end - start)
            # all_len[end - start] += 1

            self.data[i] = self.data[i][start:end]
            # self.data[i] = self.data[i] - np.average(self.data[i])
            self.data[i] = self.data[i]

            self.data[i] = np.pad(self.data[i], (64, 4480 - len(self.data[i])), mode='constant')
            for j in range(len(sin_time)):
                sin_time[j] = [sin_time[j][0] + 64, sin_time[j][1] + 64]
                sin_r_peak[j] = sin_r_peak[j] + 64
            self.time[i] = sin_time
            self.r_peak[i] = sin_r_peak

        all_len.sort()
        self.time = [self.time[i] for i in range(len(self.time)) if i not in remove]
        self.data = [self.data[i] for i in range(len(self.data)) if i not in remove]
        self.label = [self.label[i] for i in range(len(self.label)) if i not in remove]
        self.r_peak = [self.r_peak[i] for i in range(len(self.r_peak)) if i not in remove]


