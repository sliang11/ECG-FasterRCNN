from sklearn.preprocessing import scale
import scipy.signal as sig
from tensecond import tensecond
import numpy as np


class prerprocessor(tensecond):
    def filter_data(self):
        for i in range(len(self.data)):
            data = self.data[i].copy()
            b, a = sig.butter(3, 0.25, 'lowpass')
            d, c = sig.butter(3, 0.0015, 'highpass')
            fil_data = sig.filtfilt(b, a, data)
            fil_data = sig.filtfilt(d, c, fil_data)
            average = sum(fil_data) / len(fil_data)
            data = fil_data - average
            self.data[i] = data

