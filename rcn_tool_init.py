import torch
from abc import abstractclassmethod
from config import cfg
from tool.loss.focalloss import FocalLoss


class rcn_tool_init():
    def __init__(self, clsweight=None):
        self.cross_entropy = None
        self.loss_box = None
        self.detector = False
        # self.loss = my_loss()
        clsweight = [3, 0.5, 1, 0.5, 1]
        # if clsweight != None:
        #     self.loss = FocalLoss(gamma=2, alpha=clsweight)
        self.loss = FocalLoss(gamma=2, alpha=clsweight, size_average=False)
        self.save_dict = {}
        self.save_dict.setdefault("data", [])
        self.save_dict.setdefault("predict", [])
        self.save_dict.setdefault("label", [])

    @abstractclassmethod
    def pre_gt_match(self, proposal, gt_boxes, flag=0):
        pass

    @abstractclassmethod
    def pre_gt_match_uniform(self, proposal, gt_box, training=True, params=None):
        pass

    @abstractclassmethod
    def cal_loss2(self, cls_score, label):
        pass

    @abstractclassmethod
    def cal_loss(self, cls_score, box_pred, label, predict_offset, cls_weight, cor_weight):
        pass

    @abstractclassmethod
    def roi_pooling_cuda(self, features, proposal, label=None, stride=cfg.feature_stride, pool=None, batch=False):
        pass

    @abstractclassmethod
    def box_select_final(self, predict_box, score, default_box):
        pass
