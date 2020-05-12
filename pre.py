

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from data.config2 import cfg
from config import cfg

class pre(nn.Module):
    def __init__(self):
        super(pre, self).__init__()

        self.fourier = False

        self.c1 = nn.Conv1d(128, 1024, 1, bias=False)
        self.c2 = nn.Conv1d(256, 1024, 1, bias=False)
        self.c3 = nn.Conv1d(512, 1024, 1, bias=False)
        self.c4 = nn.Conv1d(1024, 1024, 1, bias=False)

    def forward(self, f1, f2, f3, f4):


        f11 = self.c1(f1)
        f22 = self.c2(f2)
        f33 = self.c3(f3)
        f44 = self.c4(f4)
        x = torch.cat([f11, f22, f33, f44], -1)

        return x
