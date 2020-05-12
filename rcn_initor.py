# from train.eval_util import *
from backbone_ import weights_init, get_paprams, adjust_learning_rate, adjust_learning_rate2
from torch.nn import DataParallel
from torch import load

match = {6: '0', 5: '1', 4: '2', 3: '0', 2: '1', 1: '2', 7: '2', 8: '0', 9: '1', 10: '2', 11: '0'}
from RPN import RPN
# from RCN import ROI
from pre import pre as pre
from RCN import ROI as roi
from config import cfg
from rcn_tool_c import rcn_tool_c
from rpn_tool_d import rpn_tool_d
from backbone_ import mb
import os


class rcn_initor():
    def __init__(self):
        # para.device_ids = [0, 1, 2]

        self.pool_size = [8, 4, 2, 1]
        # os.environ["CUDA_VISIBLE_DEVICES"] = match.get(self.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        self.tool = rpn_tool_d()
        self.tool2 = rcn_tool_c()
        path_base = os.path.join(os.getcwd(), 'base_a1_max.p')
        path_RPN = os.path.join(os.getcwd(), 'rpn_a1_max.p')

        print(path_base)
        print(path_RPN)
        from tool.batch.roi_layers import ROIPool
        self.pool4 = ROIPool(self.pool_size[3], 1 / 64)
        self.pool3 = ROIPool(self.pool_size[2], 1 / 32)
        self.pool2 = ROIPool(self.pool_size[1], 1 / 16)
        self.pool1 = ROIPool(self.pool_size[0], 1 / 8)
        self.pre = pre().cuda()
        self.ROI = roi().cuda()
        self.RPN = RPN().cuda().eval()

        self.features = mb().cuda().eval()
        tmp = load(path_RPN)
        self.RPN.load_state_dict(tmp)
        tmp = load(path_base)
        self.features.load_state_dict(tmp)
        self.features = DataParallel(self.features, device_ids=[0])
        self.RPN = DataParallel(self.RPN, device_ids=[0])
        self.ROI = DataParallel(self.ROI, device_ids=[0])
        self.pre = DataParallel(self.pre, device_ids=[0])

        get_paprams(self.features)
        get_paprams(self.RPN)
        get_paprams(self.ROI)
        self.ROI.apply(weights_init)
        self.batch = True
        self.flag = 3

        # self.save_dict = {}
        # self.save_dict.setdefault("data", [])
        # self.save_dict.setdefault("predict", [])
        # self.save_dict.setdefault("label", [])
