from multiprocessing import Process
from config import cfg
# from train.eval_util import *
# from tool.box_utils import jaccard, nms
# from backbone.ROI_10s import proposal_loss, ROI
# from tool.box_transform import offset_to_box
import torch.nn.functional as F
from rcn_initor import rcn_initor
from rcn_tool_c import rcn_tool_c
from rpn_tool_d import rpn_tool_d
from imblearn.metrics import specificity_score
import torch
import sklearn.metrics as metrics
class eval_2sg(rcn_initor):
    def sg2(self, data_loader, params):
        pass