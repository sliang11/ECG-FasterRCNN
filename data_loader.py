from data_corrector import corrector
from torch.utils.data import Dataset, DataLoader
import random
import pickle


class loader(Dataset):
    def __init__(self, path=None, params=None, index=None):
        mode = params['mode']
        a = corrector()
        a.process(path)
        self.data = a.data
        self.label = a.label
        self.window = a.time
        self.r_peak = a.r_peak
        total = len(self.label)

        if index == None:
            total = len(self.label)
            index = list(range(total))
            random.shuffle(index)
            self.index = index
        else:
            self.index = index

        if mode == 'training':
            tmp = self.index[:int(0.7 * total)]
            self.data = [self.data[i] for i in tmp]
            self.label = [self.label[i] for i in tmp]
            self.window = [self.window[i] for i in tmp]
            self.r_peak = [self.r_peak[i] for i in tmp]
        elif mode == 'test':
            tmp = self.index[-int(0.2 * total):]
            self.data = [self.data[i] for i in tmp]
            self.label = [self.label[i] for i in tmp]
            self.window = [self.window[i] for i in tmp]
            self.r_peak = [self.r_peak[i] for i in tmp]
        elif mode == 'eval':
            tmp = self.index[-int(0.3 * total):-int(0.2 * total)]
            self.data = [self.data[i] for i in tmp]
            self.label = [self.label[i] for i in tmp]
            self.window = [self.window[i] for i in tmp]
            self.r_peak = [self.r_peak[i] for i in tmp]

    def __getitem__(self, item):
        return self.data[item], self.window[item], self.label[item], self.r_peak[item], item

    def __len__(self):
        return len(self.data)
