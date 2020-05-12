import sys
from collections import OrderedDict

import time
import sys
import os

sys.path.append(os.getcwd())
from torch import save, load
from torch.optim import SGD, Adam, Adadelta
from torch.utils.data import Dataset, DataLoader
import torch
from data_loader import loader
from detection_collate_mit import call_back
from rpn_tool_d import rpn_tool_d
from backbone_ import get_paprams, adjust_learning_rate, adjust_learning_rate2

from eval_1sg import rpn_evalor


class model(rpn_evalor):

    def train_stage_1(self):
        data_set = loader(None, {"seed": 10, "mode": "training"})
        data_set_test = loader(None, {"seed": 10, "mode": "test"}, data_set.index)
        data_set_eval = loader(None, {"seed": 10, "mode": "eval"}, data_set.index)
        data_loader = DataLoader(data_set, self.batch, True, collate_fn=call_back.detection_collate_RPN, num_workers=0)
        data_loader_test = DataLoader(data_set_test, self.batch, False, collate_fn=call_back.detection_collate_RPN,
                                      num_workers=0, )

        # optim = Adadelta(self.RPN.parameters(), lr=lr1, weight_decay=1e-5)
        optim = Adadelta(self.RPN.parameters(), lr=self.lr1, weight_decay=1e-5)

        tool = rpn_tool_d()
        start_time = time.time()
        # print(optim.state_dict())
        for epoch in range(1500):
            runing_losss = 0.0
            cls_loss = 0
            coor_loss = 0

            for data in data_loader:
                y = data[1]
                x = data[0].cuda()

                optim.zero_grad()
                with torch.no_grad():
                    x1, x2, x3, x4 = self.features(x)
                predict_confidence, box_predict = self.RPN(x1, x2, x3, x4)
                cross_entropy, loss_box = tool.get_proposal(predict_confidence, box_predict, y)
                loss_total = cross_entropy + loss_box
                loss_total.backward()
                optim.step()
                runing_losss += loss_total.item()
                cls_loss += cross_entropy.item()
                coor_loss += loss_box.item()
            end_time = time.time()
            # self.vis.line(np.asarray([cls_loss, coor_loss]).reshape(1, 2),
            #               np.asarray([epoch] * 2).reshape(1, 2), win="loss-epoch", update='append',
            #               opts=dict(title='loss', legend=['cls_loss', 'cor_loss']))
            print("epoch:{a}: loss:{b:.4f} spend_time:{c:.4f} cls:{d:.4f} cor{e:.4f} date:{ff}".format(a=epoch,
                                                                                                       b=runing_losss,
                                                                                                       c=int(
                                                                                                           end_time - start_time),
                                                                                                       d=cls_loss,
                                                                                                       e=coor_loss,
                                                                                                       ff=time.asctime()))
            start_time = end_time

            # if self.add_eval:
            #     p = self.RPN_eval(self,data_loader_eval, epoch, eval=True, seed=self.seed)
            self.RPN_eval(data_loader_test, {"epoch": epoch})

            save(self.RPN.module.state_dict(),
                 os.path.join(os.getcwd(), str(epoch) + 'rpn_a1.p'))
            save(self.RPN.module.state_dict(),
                 os.path.join(os.getcwd(), str(epoch) + 'base_a1.p'))

            if epoch % 10 == 0 and epoch > 0:
                adjust_learning_rate(optim, 0.9, epoch, 50, 0.3)
            # save(self.RPN.state_dict(), para.RPN.save_path + str(_) + str("%.4f" % (runing_losss)))
        # save(self.RPN.module.state_dict(), para.stage1.save_path1)


if __name__ == '__main__':
    a = model()
    a.train_stage_1()
    # a.test(a)
