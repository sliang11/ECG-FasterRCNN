from collections import OrderedDict
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.getcwd())
# from mobilenet_ssd import MobileNetV2 as mobilenet

from tool.loss.smooth import smooth_focal_weight
from tool.loss.mixup import mixup_data, mixup_criterion
import torch.nn.functional as F
import sklearn.metrics as metrics

import time
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD, Adam, Adadelta
from torch import save, load
import sklearn.metrics as metrics
from backbone_ import *
from data_ import *
from torch.nn import DataParallel
import numpy as np
import os
from tool.loss.focalloss import FocalLoss

# from data.classifier_loader_align import my_dataset as my_dataset_aligned

# device = torch.device("cuda:0")
all_data = []
s = "python3 {}".format(os.getcwd(), 'mit.py')
os.system(s)

from loader1 import my_dataset as my_dataset_10s_smote


class model():
    def __int__(self):
        pass

    def train(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        device_ids = [0]
        self.classifier = classifier()
        get_paprams(self.classifier)
        get_paprams(self.classifier.base)
        # data_set_eval = my_dataset(eval=True)
        # data_set = my_dataset_10s()
        # data_set_test = my_dataset_10s()
        data_set = my_dataset_10s_smote()
        data_set_test = my_dataset_10s_smote(test=True, all_data=data_set.all_data, all_label=data_set.all_label,
                                             index_=data_set.index)
        # data_set_eval = my_dataset_10s(eval=True)
        # data_set_combine = my_dataset(combine=True)
        batch = 300
        totoal_epoch = 2000
        print('batch:{}'.format(batch))
        # self.evaluation = evaluation
        data_loader = DataLoader(data_set, batch, shuffle=True, collate_fn=detection_collate)
        data_loader_test = DataLoader(data_set_test, batch, False, collate_fn=detection_collate)
        # data_loader_eval = DataLoader(data_set_eval, batch, False, collate_fn=detection_collate)
        self.classifier = self.classifier.cuda()
        self.classifier = DataParallel(self.classifier, device_ids=device_ids)
        optim = Adadelta(self.classifier.parameters(), 0.1, 0.9, weight_decay=1e-5)

        self.cretion = smooth_focal_weight()

        self.classifier.apply(weights_init)
        start_time = time.time()
        count = 0
        epoch = -1
        while 1:
            epoch += 1
            runing_losss = [0] * 5
            for data in data_loader:
                loss = [0] * 5
                y = data[1].cuda()
                x = data[0].cuda()
                optim.zero_grad()

                weight = torch.Tensor([0.5, 2, 0.5, 2]).cuda()

                inputs, targets_a, targets_b, lam = mixup_data(x, y)
                predict = self.classifier(x)
                ############################3

                loss_func = mixup_criterion(targets_a, targets_b, lam, weight)
                loss5 = loss_func(self.cretion, predict[0])
                loss4 = loss_func(self.cretion, predict[1]) * 0.4
                loss3 = loss_func(self.cretion, predict[2]) * 0.3
                loss2 = loss_func(self.cretion, predict[3]) * 0.2
                loss1 = loss_func(self.cretion, predict[4]) * 0.1

                tmp = loss5 + loss4 + loss3 + loss2 + loss1

                # tmp = sum(loss)
                tmp.backward()
                optim.step()
                for i in range(5):
                    # runing_losss[i] += (loss[i].item())
                    runing_losss[i] += (tmp.item())

                count += 1
                # torch.cuda.empty_cache()
            end_time = time.time()
            print(
                "epoch:{a}: loss:{b} spend_time:{c} time:{d}".format(a=epoch, b=sum(runing_losss),
                                                                     c=int(end_time - start_time),
                                                                     d=time.asctime()))
            start_time = end_time

            # vis.line(np.asarray([optim.param_groups[0]['lr']]), np.asarray([epoch]), win="lr", update='append',
            #          opts=dict(title='lr'))
            # if (epoch > 20):
            #     runing_losss = np.asarray(runing_losss).reshape(1, 5)

            # vis.line(runing_losss,
            #          np.asarray([epoch] * 5).reshape(1, 5), win="loss-epoch", update='append',
            #          opts=dict(title='loss', legend=['loss1', 'loss2', 'loss3', 'loss4', 'loss5', 'loss6']))
            save(self.classifier.module.base.state_dict(),
                 str(epoch) + 'base_c2.p')
            save(self.classifier.module.state_dict(),
                 str(epoch) + 'base_all_c2.p')
            # print('eval:{}'.format(time.asctime(time.localtime(time.time()))))
            self.classifier.eval()
            # self.evaluation(self.classifier, data_loader_eval)
            # print('test:{}'.format(time.asctime(time.localtime(time.time()))))
            # self.evaluation(self.classifier, data_loader_eval, epoch)
            self.evaluation(self.classifier, data_loader_test, epoch)
            # self.evaluation(self.classifier, data_loader, epoch)

            # print('combine:{}'.format(time.asctime(time.localtime(time.time()))))
            # evaluation(self.classifier, data_loader_combine)
            self.classifier.train()
            if epoch % 10 == 0:
                adjust_learning_rate(optim, 0.9, epoch, totoal_epoch, 0.1)
        # print('eval')
        # self.evaluation(self.classifier, data_loader_eval)
        # self.evaluation(self.classifier, data_loader_test, 500)

    def evaluation(self, classifier, data_loader_test, epoch):
        # classifier.eval()
        all_predict = [[], [], [], [], []]
        all_ground = []
        with torch.no_grad():
            for data in data_loader_test:
                y = data[1].cuda()
                x = data[0].cuda()
                predict_list = classifier(x)
                # predict = F.softmax(predict_list[0], 1)
                for i in range(5):
                    predict, index = torch.max(predict_list[i], 1)
                    all_predict[i].extend(index.tolist())

                all_ground.extend(y.tolist())
        # weight = [0.3, 1, 0.3, 1]
        # weight = [i for i in all_ground]
        print("Accuracy:{}".format(metrics.accuracy_score(all_ground, all_predict[0])))
        print('precesion:{}'.format(metrics.precision_score(all_ground, all_predict[0], average=None)))
        print('recall:{}'.format(metrics.recall_score(all_ground, all_predict[0], average=None)))
        print('f-score:{}'.format(metrics.f1_score(all_ground, all_predict[0], average=None)))
        print("{}".format(metrics.confusion_matrix(all_ground, all_predict[0])))
        for i in range(5):
            tmp = metrics.accuracy_score(all_ground, all_predict[i])
            tmp2 = metrics.precision_score(all_ground, all_predict[i], average=None)
            tmp3 = metrics.recall_score(all_ground, all_predict[i], average=None)
            tmp4 = metrics.f1_score(all_ground, all_predict[i], average=None)
            print("Accuracy:{}".format(tmp))
            print('precesion:{}'.format(tmp2))
            print('recall:{}'.format(tmp3))
            print('f-score:{}'.format(tmp4))

    def test(self):
        self.classifier = classifier()
        self.classifier = self.classifier.cuda()
        data_set = my_dataset_10s_smote(test=True)
        data_loader_test = DataLoader(data_set, 300, False, collate_fn=detection_collate)
        all_predict = []
        all_ground = []
        self.classifier.eval()
        self.classifier.base.eval()
        total = 0
        with torch.no_grad():
            for data in data_loader_test:
                y = data[1].cuda()
                x = data[0].cuda()
                every_len = data[2]
                max_len = data[3]
                predict = self.classifier(x, every_len, max_len)[0]
                # predict = F.softmax(predict, 1)
                predict, index = torch.max(predict, 1)
                # total += predict.sum().item()
                ########
                ###
                all_predict.extend(list(index.cpu().numpy()))
                all_ground.extend(list(y.cpu().numpy()))
        # print(sum(all_predict))
        # print(sum(all_ground))
        print(metrics.precision_score(all_ground, all_predict, average=None))
        print(metrics.recall_score(all_ground, all_predict, average=None))
        print(metrics.f1_score(all_ground, all_predict, average=None))
        print(metrics.confusion_matrix(all_ground, all_predict))
        # pass

    # def evaluation(self, data_loader_test, epoch):
    #     para.smote = False
    #     self.classifier.eval()
    #     # self.front.eval()
    #     # this_classifier = classifier().cpu()
    #     # tmp = load(save_path+ str(epoch) + 'all_regular')
    #     # this_classifier.load_state_dict(tmp)
    #     # this_classifier.eval()
    #     all_predict = []
    #     all_ground = []
    #     with torch.no_grad():
    #         for data in data_loader_test:
    #             y = data[1].cuda()
    #             x = data[0].cuda()
    #             # x = self.front(x)
    #             every_len = data[2]
    #             predict, predict1, predict2, predict3, predict4 = self.classifier(x)
    #             predict = F.softmax(predict, 1)
    #             predict, index = torch.max(predict, 1)
    #             all_predict.extend(list(index.cpu().numpy()))
    #             all_ground.extend(list(y.cpu().numpy()))
    #     print('precesion:{}'.format(metrics.precision_score(all_ground, all_predict, average=None)))
    #     print('recall:{}'.format(metrics.recall_score(all_ground, all_predict, average=None)))
    #     print('f-score:{}'.format(metrics.f1_score(all_ground, all_predict, average=None)))
    #     self.classifier.train()
    # self.front.eval()

    # elif isinstance(m, nn.BatchNorm1d):
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.fill_(0)


if __name__ == '__main__':
    a = model()
    a.train()
    # a.test()
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # torch.set_default_tensor_type('torch.FloatTensor')
