import torch
from tool.loss.focalloss import FocalLoss
from config import cfg
from layer.generate_anchors import gene_default
from abc import abstractclassmethod
import torch.nn.functional as F
class rpn_initor():
    def __init__(self):
        # self.ce = nn.CrossEntropyLoss()
        self.train_mode = True
        # self.loss = my_loss()
        self.loss = FocalLoss(2, alpha=[0.25, 1])
        self.gene_original_box(cfg.feature_length)
        # 56-mit     47-mit_second

    def gene_original_box(self, K=cfg.feature_length):

        self.default_box = gene_default()
        self.default_box = self.default_box.generate_anchors().cuda()
        length = K

        original_size = (torch.arange(0, length) * cfg.feature_stride).view(-1, 1)
        original_size = torch.cat([original_size, original_size], 1).type(torch.FloatTensor).cuda()

        A = cfg.num_box
        # K = para.feature_length
        anchor = self.default_box.unsqueeze(0).expand(K, A, 2) + original_size.unsqueeze(1).expand(K, A, 2)
        anchor = anchor.view(K * A, 2)
        self.default_box = anchor
    @abstractclassmethod
    def build_loss(self, predict_confidence, box_predict, default_label, default_gt_offset):
        pass
    @abstractclassmethod
    def get_proposal(self, predict_confidence, box_predict, gt_boxes, test=False):
        pass