import torch
import torch.nn.functional as F

import torch.nn as nn
import math
from config import cfg
import torch.nn.init as init
def xavier(param):
    init.xavier_uniform_(param)


def adjust_learning_rate(optimizer, gamma, step, decay_step, base_lr):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = base_lr * (gamma ** (step / decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate2(optimizer, gamma, step, decay_step, base_lr):
    for i in range(len(optimizer.param_groups)):
        optimizer.param_groups[i]['lr'] = base_lr[i] * (gamma ** (step / decay_step))


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        xavier(m.weight.data)
    elif isinstance(m, nn.Linear):
        xavier(m.weight.data)


def get_paprams(classifier):
    params = list(classifier.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print(str(k))
class conv_bn(nn.Module):
    def __init__(self, inp, oup, stride):
        super(conv_bn, self).__init__()
        self.conv = nn.Conv1d(inp, oup, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm1d(oup)
        self.a1 = nn.ELU(inplace=True)
        self.p1 = nn.MaxPool1d(2, 2)

        # return nn.Sequential(
        #      if stride == 1 else nn.Conv1d(inp, oup, 2, stride, 0,
        #                                                                                 bias=False),
        #     nn.BatchNorm1d(oup),
        #     nn.ELU(inplace=True)
        # )

    def forward(self, x):
        x = self.conv(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        # pw
        self.conv1 = nn.Conv1d(inp, hidden_dim, 1, 1, 0, bias=False)
        self.b1 = nn.BatchNorm1d(hidden_dim)
        self.a1 = nn.ELU(inplace=True)
        # dw
        if stride == 1:
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 5, stride, 2, groups=hidden_dim,
                                   bias=False)
        else:

            self.conv2 = nn.MaxPool1d(2, 2)



        self.b2 = nn.BatchNorm1d(hidden_dim)
        self.a2 = nn.ELU(inplace=True)
        # pw-linear
        self.conv3 = nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.b3 = nn.BatchNorm1d(oup)
        if not self.use_res_connect:#2019.6.11
            self.res = nn.Sequential(nn.Conv1d(inp, oup, 1, bias=False), nn.AvgPool1d(2, 2))

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.b1(x1)
        x1 = self.a1(x1)
        x1 = self.conv2(x1)
        x1 = self.b2(x1)
        x1 = self.a2(x1)
        x1 = self.conv3(x1)
        x1 = self.b3(x1)
        if self.use_res_connect:
            return x + x1
        else:
            return x1 + self.res(x)
            # return x1




class mb(nn.Module):
    def __init__(self, in_channel=1, n_class=1000, input_size=224, width_mult=1.):
        super(mb, self).__init__()
        block = InvertedResidual
        input_channel = 32
        # interverted_residual_setting = cfg.mb_ori_ltdb_set
        # input_channel = int(input_channel * width_mult)
        # self.features = [conv_bn(in_channel, input_channel, 2)]

        self.conv1 = conv_bn(in_channel, input_channel, 2)
        self.conv1p = block(input_channel, input_channel, 1, 2)

        self.conv2 = block(input_channel, cfg.mb_ori_ltdb_set[0][1], 2, 2)
        self.conv2p = block(cfg.mb_ori_ltdb_set[0][1], cfg.mb_ori_ltdb_set[0][1], 1, 2)

        self.conv3 = block(cfg.mb_ori_ltdb_set[0][1], cfg.mb_ori_ltdb_set[1][1], 2, 2)
        self.conv4 = block(cfg.mb_ori_ltdb_set[1][1], cfg.mb_ori_ltdb_set[1][1], 1, 2)

        self.conv5 = block(cfg.mb_ori_ltdb_set[1][1], cfg.mb_ori_ltdb_set[2][1], 2, 3)
        self.conv6 = block(cfg.mb_ori_ltdb_set[2][1], cfg.mb_ori_ltdb_set[2][1], 1, 3)
        self.conv7 = block(cfg.mb_ori_ltdb_set[2][1], cfg.mb_ori_ltdb_set[2][1], 1, 3)

        self.conv8 = block(cfg.mb_ori_ltdb_set[2][1], cfg.mb_ori_ltdb_set[3][1], 2, 3)
        self.conv9 = block(cfg.mb_ori_ltdb_set[3][1], cfg.mb_ori_ltdb_set[3][1], 1, 3)
        self.conv11 = block(cfg.mb_ori_ltdb_set[3][1], cfg.mb_ori_ltdb_set[3][1], 1, 3)

        self.conv12 = block(cfg.mb_ori_ltdb_set[3][1], cfg.mb_ori_ltdb_set[4][1], 2, 2)
        self.conv14 = block(cfg.mb_ori_ltdb_set[4][1], cfg.mb_ori_ltdb_set[4][1], 1, 2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1p(x)
        #
        x = self.conv2(x)
        x = self.conv2p(x)
        #
        x = self.conv3(x)
        x1 = self.conv4(x)

        x = self.conv5(x1)
        x = self.conv6(x)
        x2 = self.conv7(x)

        x = self.conv8(x2)
        x = self.conv9(x)
        x3 = self.conv11(x)

        x = self.conv12(x3)
        x4 = self.conv14(x)

        return x1, x2, x3, x4


class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()

        self.base = mb()

        self.fc1 = nn.Linear(cfg.mb_ori_ltdb_set[-4][1], 4, bias=False)
        self.fc2 = nn.Linear(cfg.mb_ori_ltdb_set[-3][1], 4, bias=False)
        self.fc3 = nn.Linear(cfg.mb_ori_ltdb_set[-2][1], 4, bias=False)
        self.fc4 = nn.Linear(cfg.mb_ori_ltdb_set[-1][1], 4, bias=False)
        self.fc = nn.Linear(
            cfg.mb_ori_ltdb_set[-4][1] + cfg.mb_ori_ltdb_set[-2][1] + cfg.mb_ori_ltdb_set[-3][1] + cfg.mb_ori_ltdb_set[-1][1], 4, True)
        self.dr = nn.Dropout(0.5)

    def forward(self, input, every_len=1, max_len=1):
        x1, x2, x3, x4 = self.base(input)
        x1 = F.avg_pool1d(x1, 2, 2) + F.max_pool1d(x1, 2, 2)
        x1 = F.avg_pool1d(x1, 2, 2) + F.max_pool1d(x1, 2, 2)
        x1 = F.max_pool1d(x1, 2, 2)
        x2 = F.avg_pool1d(x2, 2, 2) + F.max_pool1d(x2, 2, 2)
        x2 = F.max_pool1d(x2, 2, 2)
        x3 = F.max_pool1d(x3, 2, 2)
        # x11 = self.conv1(x1)
        # x22 = self.conv2(x2)
        # x33 = self.conv3(x3)
        # x44 = x4
        # x44 = F.dropout(x44, 0.5)
        # x33 = F.dropout(x33, 0.5)
        # x22 = F.dropout(x22, 0.5)
        # x11 = F.dropout(x11, 0.5)
        # x = x11 + x22 + x33 + x44
        x = torch.cat([x1, x2, x3, x4], 1)
        x = F.dropout(x, 0.5)
        # x = self.cc(x)
        x = torch.mean(x, 2)
        # x = self.dr(x)
        x = self.fc(x)
        # x = F.softmax(x, 1)
        x1 = F.dropout(x1, 0.5)
        x2 = F.dropout(x2, 0.5)
        x3 = F.dropout(x3, 0.5)
        x4 = F.dropout(x4, 0.5)

        x1 = torch.mean(x1, 2)
        x2 = torch.mean(x2, 2)
        x3 = torch.mean(x3, 2)
        x4 = torch.mean(x4, 2)
        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x3 = self.fc3(x3)
        x4 = self.fc4(x4)

        return x, x1, x2, x3, x4
