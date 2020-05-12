import sys
import os

sys.path.append(os.getcwd())
import time

from torch import save, load
from torch.optim import SGD, Adam, Adadelta
from torch.utils.data import Dataset, DataLoader
# from backbone.RPN_10s import RPN
# from backbone.RPN_10s import tool as T
from data_loader import loader
from config import cfg
from detection_collate_mit import call_back
# from train.eval_util import *
from rcn_tool_c import rcn_tool_c
from base_process import base_process
# from backbone.backbone_ import weights_init, get_paprams, adjust_learning_rate, adjust_learning_rate2
from backbone_ import *
import torch
from RCN import ROI as roi
import sklearn.metrics as metrics
from torch.nn import DataParallel
from imblearn.metrics import specificity_score


class model(base_process):
    def train_stage_2(self):

        batch = 240
        lr1 = 0.15

        data_set = loader(os.path.join(os.getcwd(), 'data_2'), {"mode": "training"})
        data_set_test = loader(os.path.join(os.getcwd(), 'data_2'),{"mode": "test"}, data_set.index)
        data_set_eval = loader(os.path.join(os.getcwd(), 'data_2'),{"mode": "eval"}, data_set.index)

        data_loader = DataLoader(data_set, batch, True, collate_fn=call_back.detection_collate_RPN)
        data_loader_test = DataLoader(data_set_test, batch, False, collate_fn=call_back.detection_collate_RPN)
        data_loader_eval = DataLoader(data_set_eval, batch, False, collate_fn=call_back.detection_collate_RPN)

        # optim = Adadelta(self.ROI.parameters(), lr=lr1, weight_decay=1e-5)
        start_time = time.time()
        optim_a = Adadelta([{'params': self.pre.parameters()},
                            {'params': self.ROI.parameters()}], lr=0.15, weight_decay=1e-5)
        cfg.test = False
        count = 0
        for epoch in range(200):
            runing_losss = 0.0
            cls_loss = 0
            coor_loss = 0
            cls_loss2 = 0
            coor_loss2 = 0
            count += 1
            # base_time = RPN_time = ROI_time = nms_time = pre_gt = loss_time = linear_time = 0
            for data in data_loader:
                y = data[1]
                x = data[0].cuda()
                peak = data[2]
                num = data[3]
                optim_a.zero_grad()

                with torch.no_grad():
                    if self.flag >= 2:
                        result = self.base_process(x, y, peak)
                        feat1 = result['feat_8']
                        feat2 = result['feat_16']
                        feat3 = result['feat_32']
                        feat4 = result['feat_64']
                        label = result['label']
                        loss_box = result['loss_box']
                        cross_entropy = result['cross_entropy']

                cls_score = self.pre(feat1, feat2, feat3, feat4)
                cls_score = self.ROI(cls_score)

                cross_entropy2 = self.tool2.cal_loss2(cls_score, label)

                loss_total = cross_entropy2
                loss_total.backward()
                optim_a.step()
                runing_losss += loss_total.item()
                cls_loss2 += cross_entropy2.item()
                cls_loss += cross_entropy.item()
                coor_loss += loss_box.item()
            end_time = time.time()
            torch.cuda.empty_cache()
            print(
                "epoch:{a} time:{ff}: loss:{b:.4f} cls:{d:.4f} cor{e:.4f} cls2:{f:.4f} cor2:{g:.4f} date:{fff}".format(
                    a=epoch,
                    b=runing_losss,
                    d=cls_loss,
                    e=coor_loss,
                    f=cls_loss2,
                    g=coor_loss2, ff=int(end_time - start_time),
                    fff=time.asctime()))
            # if epoch % 10 == 0:
            #     adjust_learning_rate(optim, 0.9, epoch, 50, lr1)
            p = None

            # if epoch % 2 == 0:
            #     print("test result")
            # save(self.RPN.module.state_dict(),
            #      os.path.join(os.getcwd(), str(epoch) + 'rpn_a2.p'))
            # save(self.RPN.module.state_dict(),
            #      os.path.join(os.getcwd(), str(epoch) + 'base_a2.p'))
            start_time = end_time
        all_data = []
        all_label = []
        for data in data_loader:
            y = data[1]
            x = data[0].cuda()
            num = data[3]
            peak = data[2]
            with torch.no_grad():
                if self.flag >= 2:
                    result = self.base_process_2(x, y, peak)
                    data_ = result['x']
                    label = result['label']
                    loss_box = result['loss_box']
                    cross_entropy = result['cross_entropy']
                    all_data.extend(data_.cpu())
                    all_label.extend(label.cpu())
        for data in data_loader_eval:
            y = data[1]
            x = data[0].cuda()
            num = data[3]
            peak = data[2]
            with torch.no_grad():
                if self.flag >= 2:
                    result = self.base_process_2(x, y, peak)
                    data_ = result['x']
                    label = result['label']
                    loss_box = result['loss_box']
                    cross_entropy = result['cross_entropy']
                    all_data.extend(data_.cpu())
                    all_label.extend(label.cpu())
        for data in data_loader_test:
            y = data[1]
            x = data[0].cuda()
            num = data[3]
            peak = data[2]
            with torch.no_grad():
                if self.flag >= 2:
                    result = self.base_process_2(x, y, peak)
                    data_ = result['x']
                    label = result['label']
                    loss_box = result['loss_box']
                    cross_entropy = result['cross_entropy']
                    all_data.extend(data_.cpu())
                    all_label.extend(label.cpu())

        all_data = torch.stack(all_data, 0).numpy()
        all_label = torch.LongTensor(all_label).numpy()
        from imblearn.over_sampling import SMOTE
        fun = SMOTE()
        all_data, all_label = fun.fit_resample(all_data, all_label)
        total = len(all_label)
        training_label = all_label[:int(0.7 * total)]
        training_data = all_data[:int(0.7 * total)]

        test_label = all_label[-int(0.2 * total):]
        test_data = all_data[-int(0.2 * total):]
        count = 0
        self.ROI = roi().cuda()
        self.ROI = DataParallel(self.ROI, device_ids=[0])
        self.ROI.apply(weights_init)

        optim_b = Adadelta(self.ROI.parameters(), lr=0.15, weight_decay=1e-5)
        for epoch in range(1200):
            runing_losss = 0.0
            cls_loss = 0
            coor_loss = 0
            cls_loss2 = 0
            coor_loss2 = 0
            count += 1
            optim_b.zero_grad()
            optim_a.zero_grad()

            # base_time = RPN_time = ROI_time = nms_time = pre_gt = loss_time = linear_time = 0
            for j in range(int(len(training_label) / 240)):
                data_ = torch.Tensor(training_data[j * 240:j * 240 + 240]).view(240, 1024, 15).cuda()
                label_ = torch.LongTensor(training_label[j * 240:j * 240 + 240]).cuda()
                optim_b.zero_grad()

                cls_score = self.ROI(data_)
                cross_entropy2 = self.tool2.cal_loss2(cls_score, label_)

                loss_total = cross_entropy2
                loss_total.backward()
                optim_b.step()
                runing_losss += loss_total.item()
                cls_loss2 += cross_entropy2.item()
                cls_loss += cross_entropy.item()
                coor_loss += loss_box.item()
            end_time = time.time()
            torch.cuda.empty_cache()
            print(
                "epoch:{a} time:{ff}: loss:{b:.4f} cls:{d:.4f} cor{e:.4f} cls2:{f:.4f} cor2:{g:.4f} date:{fff}".format(
                    a=epoch,
                    b=runing_losss,
                    d=cls_loss,
                    e=coor_loss,
                    f=cls_loss2,
                    g=coor_loss2, ff=int(end_time - start_time),
                    fff=time.asctime()))
            if epoch % 10 == 0 and epoch > 0:
                adjust_learning_rate(optim_b, 0.9, epoch, 50, 0.3)

            p = None
            self.eval_(test_data, test_label)
            # self.ROI_eval(data_loader_eval, {"epoch": epoch})

            start_time = end_time
        print('finish')

    def eval_(self, data, label):
        self.ROI = self.ROI.eval()
        gt = []
        pre = []
        total = int(len(label) / 240)
        with torch.no_grad():
            for i in range(total):
                a = i * 240
                b = a + 240
                sin_x = torch.Tensor(data[a:b]).cuda()
                sin_x = sin_x.view(240, 1024, 15)
                sin_y = label[a:b]
                predict = self.ROI(sin_x)
                predict, index = torch.max(predict, 1)
                pre.extend(index.cpu().tolist())
                gt.extend(sin_y)
        print("ppv:{}".format(metrics.precision_score(gt, pre, average='micro')))
        print("spe:{}".format(specificity_score(gt, pre, average='micro')))
        print("sen:{}".format(metrics.recall_score(gt, pre, average='micro')))


    def base_process_2(self, x, y, peak):
        cross_entropy, loss_box = torch.ones(1), torch.ones(1)
        with torch.no_grad():
            x1, x2, x3, x4 = self.features(x)
            if self.flag == 3:
                predict_confidence, box_predict = self.RPN(x1, x2, x3, x4)
                proposal, batch_offset, batch_conf = self.tool.get_proposal(predict_confidence, box_predict,
                                                                            y, test=True)
                # save_proposal = [i.cpu().numpy() for i in proposal]
                # save_data = x.cpu().numpy()
                # save_y = [i.numpy() for i in y]
                # self.save_dict['data'].append(save_data)
                # self.save_dict['label'].append(save_y)
                # self.save_dict['predict'].append(save_proposal)

            proposal, label = self.tool2.pre_gt_match_uniform(proposal, y, training=True, params={'peak': peak})

            if 1:
                for i in range(len(proposal)):
                    tmp = torch.zeros(proposal[i].size()[0], 1).fill_(
                        i).cuda()
                    proposal[i] = torch.cat([tmp, proposal[i]], 1)
                proposal = torch.cat(proposal, 0)

            feat4, label, class_num = self.tool2.roi_pooling_cuda(x4, proposal, label=label, stride=64,
                                                                  pool=self.pool4,
                                                                  batch=True)
            feat3 = \
                self.tool2.roi_pooling_cuda(x3, proposal, stride=64, pool=self.pool3,
                                            batch=True, label=None)[
                    0]
            feat2 = \
                self.tool2.roi_pooling_cuda(x2, proposal, stride=32,
                                            pool=self.pool2,
                                            batch=True, label=None)[0]
            feat1 = \
                self.tool2.roi_pooling_cuda(x1, proposal, stride=16,
                                            pool=self.pool1,
                                            batch=True, label=None, )[0]

            x = self.pre(feat1, feat2, feat3, feat4)
            x = x.view(-1, 1024 * 15)
            if self.flag == 2:
                result = {}
                result['x'] = x
                result['label'] = label
                result['predict_offset'] = 0
                result['class_num'] = class_num
                result['batch_cor_weight'] = 0
                result['cross_entropy'] = cross_entropy
                result['loss_box'] = loss_box
                return result
            elif self.flag == 3:
                result = {}
                result['x'] = x
                result['label'] = label
                result['class_num'] = class_num
                result['cross_entropy'] = cross_entropy
                result['loss_box'] = loss_box

                return result


if __name__ == '__main__':
    a = model()
    a.train_stage_2()
    # a.test()
    # a.save_data()
    # a.smote_train()
