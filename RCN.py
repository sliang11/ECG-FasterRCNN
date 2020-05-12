

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from data.config2 import cfg
from config import cfg

class ROI(nn.Module):
    def __init__(self):
        super(ROI, self).__init__()

        self.conv1 = nn.Conv1d(1024, 1024, 3, 1, 1, bias=False)
        self.elu = nn.ELU(inplace=True)
        self.bn = nn.BatchNorm1d(1024)
        self.conv2 = nn.Conv1d(1024, 512, 3, 1, 1, bias=False)
        self.elu2 = nn.ELU(inplace=True)
        self.bn2 = nn.BatchNorm1d(512)

        self.score_fc = nn.Conv1d(512, 512, 3, 1, 1, bias=False)
        self.score_fc4 = nn.Linear(512, cfg.classes, bias=False)

    def forward(self,x):

        x = self.conv1(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = F.dropout(x, 0.5)
        x1 = self.score_fc(x)

        x1 = x1.mean(-1)
        cls_score = self.score_fc4(x1)
        cls_score = cls_score.squeeze()

        return cls_score
