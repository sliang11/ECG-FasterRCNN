from config import cfg
import torch
# from train.eval_util import *
import pickle
from box_utils import jaccard
from rpn_eval_tool import eval_tool
from rpn_eval_tool import eval_tool
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from rpn_tool_d import rpn_tool_d


class rpn_evalor(eval_tool):
    def RPN_eval(self, data_loader, params):
        self.save_dict = {}
        # self.save_dict['pre_window'] = []
        # self.save_dict['data'] = []
        # self.save_dict['ground_window'] = []
        # self.save_dict['false_window'] = []
        # self.save_dict['false_score'] = []

        self.features = self.features.eval()
        self.RPN = self.RPN.eval()
        tool2 = rpn_tool_d()
        tool2.train_mode = False
        all_proposal = []
        # seed = params['seed']
        epoch = params['epoch']

        info = dict()
        info.setdefault('gt', [])
        info.setdefault('pre', [])
        info.setdefault('tp', 0)
        info.setdefault('fp', 0)
        info.setdefault('tn', 0)
        info.setdefault('fn', 0)
        info.setdefault('gt_bin', [])
        info.setdefault('pre_bin', [])
        # if para.save_data:
        #     save_dict = {}

        for data in data_loader:
            y = data[1]
            x = data[0].cuda()
            r_peaks = data[2]
            nums = data[3]
            with torch.no_grad():
                x1, x2, x3, x4 = self.features(x)
                predict_confidence, box_predict = self.RPN(x1, x2, x3, x4)
                proposal, conf, batch_offset = tool2.get_proposal(predict_confidence, box_predict, y, test=True)
            all_proposal.append([])

            for this_batch in range(len(y)):
                this_ground = y[this_batch][:, :2].cuda()
                gt_label = y[this_batch][:, -1].type(torch.uint8).cuda()
                this_predict = proposal[this_batch]
                this_conf = conf[this_batch].cpu()
                keep3 = this_conf >= 0.5
                sin_r_peak = r_peaks[this_batch]
                # self.process_each_window(this_predict, x[this_batch].view(-1))
                if (keep3.sum().item() > 0):
                    this_predict = this_predict[keep3]
                    this_conf = this_conf[keep3]

                overlaps, union, non_overlap, non_overlap2 = jaccard(this_ground, this_predict)

                maxlap_of_pre, maxidx_of_pre = overlaps.max(0)
                maxlap_of_ground, maxidx_of_ground = overlaps.max(1)

                minlap_of_pre, minidx_of_pre = non_overlap.min(0)
                minlap_of_ground, minidx_of_ground = non_overlap2.min(1)

                params = dict()
                params.setdefault('ground_window', this_ground.cpu())
                params.setdefault('maxlap_of_ground', maxlap_of_ground)
                params.setdefault('maxidx_of_ground', maxidx_of_ground)
                params.setdefault('minlap_of_pre', minlap_of_pre)
                params.setdefault('minidx_of_pre', minidx_of_pre)

                params.setdefault('minlap_of_ground', minlap_of_ground)
                params.setdefault('minidx_of_ground', minidx_of_ground)

                params.setdefault('num', nums[this_batch])

                params.setdefault('peak', sin_r_peak)

                params.setdefault('pre_window', this_predict.cpu())
                params['save'] = self.save_dict
                params['data'] = x[this_batch].cpu().numpy()
                params.setdefault('gt_label', gt_label)
                self.first_process(info, params)

        gt = info.get("gt")
        pre = info.get("pre")
        gt_bin = info.get("gt_bin")
        pre_bin = info.get("pre_bin")

        print("acc:{}".format(accuracy_score(gt_bin, pre_bin)))
        print("precision:{}".format(precision_score(gt_bin, pre_bin)))
        print("recall:{}".format(recall_score(gt_bin, pre_bin)))
        # print("f1-score:{}".format(f1_score(gt_bin, pre_bin)))
        print("confusion:{}".format(confusion_matrix(gt_bin, pre_bin)))
        if accuracy_score(gt_bin, pre_bin) > self.max_acc:
            self.max_acc = accuracy_score(gt_bin, pre_bin)
            self.max_pre = precision_score(gt_bin, pre_bin)
            self.max_recall = recall_score(gt_bin, pre_bin)
        print("acc:{}".format(self.max_acc))
        print("precision:{}".format(self.max_pre))
        print("recall:{}".format(self.max_recall))
        # print("f1-score:{}".format(f1_score(gt_bin, pre_bin)))
        # print("tp:{a} fp:{b} fn:{c}".format(a=info.get("tp"), b=info.get("fp"), c=info.get('fn')))

        tool2.train = True
        self.RPN = self.RPN.train()
        self.features = self.features.train()

    # def process_each_window(self, window, data):
    #     for i in range(len(window)):
    #         start = int(window[i][0])
    #         end = int(window[i][1])
    #         region = data[start:end]
    #         max_point, index = torch.max(region, dim=-1)
    #         window[i][0] = start + index - 70
    #         window[i][1] = start + index + 140
