import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class RPN(nn.Module):


    def __init__(self):
        super(RPN, self).__init__()

        tmp = cfg.mb_ori_ltdb_set[-1][1] + cfg.mb_ori_ltdb_set[-2][1] + cfg.mb_ori_ltdb_set[-3][1] + cfg.mb_ori_ltdb_set[-4][1]
        self.pre_conv = nn.Conv1d(tmp, 1024, 3, 1, 1, bias=False)

        self.bn5 = nn.BatchNorm1d(1024)
        self.elu = nn.ELU(inplace=True)

        #######################
        self.score_conv5 = nn.Conv1d(1024, len(cfg.anchor_scale) * 2, 1, bias=False)  # 魔改2019.4.7

        ######################

        #############################
        self.bbox_conv1 = nn.Conv1d(1024, 512, 3, 1, 1, bias=False)
        self.bbox_conv2 = nn.Conv1d(1024, 512, 5, 1, 2, bias=False)
        self.bbox_conv3 = nn.Conv1d(1024, 512, 7, 1, 3, bias=False)
        self.bbox_conv5 = nn.Conv1d(1536, len(cfg.anchor_scale) * 2, 1, bias=False)

        ########################
        self.bn1 = nn.BatchNorm1d(cfg.mb_ori_ltdb_set[-1][1])
        self.bn2 = nn.BatchNorm1d(cfg.mb_ori_ltdb_set[-2][1])
        self.bn3 = nn.BatchNorm1d(cfg.mb_ori_ltdb_set[-3][1])
        self.bn4 = nn.BatchNorm1d(cfg.mb_ori_ltdb_set[-4][1])

    def forward(self, x1, x2, x3, x4):

        x1 = F.avg_pool1d(x1, 2, 2) + F.max_pool1d(x1, 2, 2)
        x1 = F.avg_pool1d(x1, 2, 2) + F.max_pool1d(x1, 2, 2)
        x1 = F.max_pool1d(x1, 2, 2) + F.avg_pool1d(x1, 2, 2)
        x2 = F.avg_pool1d(x2, 2, 2) + F.max_pool1d(x2, 2, 2)
        x2 = F.max_pool1d(x2, 2, 2) + F.avg_pool1d(x2, 2, 2)
        x3 = F.max_pool1d(x3, 2, 2) + F.avg_pool1d(x3, 2, 2)
        x1 = self.bn4(x1)
        x2 = self.bn3(x2)
        x3 = self.bn2(x3)
        x4 = self.bn1(x4)
        # rpn_conv = torch.cat([x1, x2, x3, x4], 1)

        rpn_conv = torch.cat([x1, x2, x3, x4], 1)
        rpn_conv = F.dropout(rpn_conv, 0.5)
        rpn_conv = self.pre_conv(rpn_conv)
        ################################
        rpn_conv = self.bn5(rpn_conv)
        rpn_conv = self.elu(rpn_conv)


        predict_confidence = self.score_conv5(rpn_conv)
        predict_confidence = predict_confidence.permute(0, 2, 1).contiguous().view(-1, rpn_conv.size()[-1] * len(
            cfg.anchor_scale), 2)  # [batch,length*3,2]


        box_predict1 = self.bbox_conv1(rpn_conv)
        box_predict2 = self.bbox_conv2(rpn_conv)
        box_predict3 = self.bbox_conv3(rpn_conv)
        box_predict = torch.cat([box_predict1, box_predict2, box_predict3], 1)
        box_predict = self.bbox_conv5(box_predict)

        ###################3

        box_predict = box_predict.permute(0, 2, 1).contiguous().view(-1, rpn_conv.size()[-1] * len(
            cfg.anchor_scale), 2)

        return predict_confidence, box_predict
