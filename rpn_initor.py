import os
from torch import load
# from backbone.backbone_ import get_paprams, adjust_learning_rate, adjust_learning_rate2
from torch.nn import DataParallel
# from config import cfg as cfg2
from RPN import RPN
from config import cfg
from data_ import *
from backbone_ import *
match = {6: '0', 5: '1', 4: '2', 3: '0', 2: '1', 1: '2', 7: '2', 8: '0', 9: '1', 10: '2', 11: '0'}
from backbone_ import mb


class rpn_initor():
    def __init__(self):

        self.lr1 = 0.15
        self.max_pre = 0
        self.max_acc = 0
        self.max_recall = 0
        self.batch = 240
        # os.environ["CUDA_VISIBLE_DEVICES"] = match[self.seed]
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'



        self.features = mb().eval()

        path = os.path.join(os.getcwd(), "base_max.p")
        tmp = load(path)
        print(path)

        self.features.load_state_dict(tmp)
        self.RPN = RPN()
        # get_paprams(self.RPN)
        self.features = self.features.cuda()
        self.RPN = self.RPN.cuda()

        self.RPN.apply(weights_init)
        self.features = DataParallel(self.features, device_ids=[0])
        self.RPN = DataParallel(self.RPN, device_ids=[0])
