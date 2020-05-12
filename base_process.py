from eval_2sg import eval_2sg
from config import cfg
import torch
from imblearn.over_sampling import SMOTE


class base_process(eval_2sg):
    def base_process(self, x, y, peak):
        cross_entropy, loss_box = torch.ones(1), torch.ones(1)
        with torch.no_grad():
            x1, x2, x3, x4 = self.features(x)
            if self.flag == 2:
                pass
            elif self.flag == 3:

                predict_confidence, box_predict = self.RPN(x1, x2, x3, x4)
                proposal, batch_offset, batch_conf = self.tool.get_proposal(predict_confidence, box_predict,
                                                                            y, test=True)
                # save_proposal = [i.cpu().numpy() for i in proposal]
                # save_data = x.cpu().numpy()
                # save_y = [i.numpy() for i in y]
                # self.save_dict['data'].append(save_data)
                # self.save_dict['label'].append(save_y)
                # self.save_dict['predict'].append(save_proposal)

            if self.flag != 3:
                pass
            else:
                proposal, label = self.tool2.pre_gt_match_uniform(proposal, y, training=True, params={'peak': peak})

            if self.batch == True:
                for i in range(len(proposal)):
                    tmp = torch.zeros(proposal[i].size()[0], 1).fill_(
                        i).cuda()
                    proposal[i] = torch.cat([tmp, proposal[i]], 1)
                proposal = torch.cat(proposal, 0)

            feat4, label, class_num = self.tool2.roi_pooling_cuda(x4, proposal, label, stride=cfg.feature_stride,
                                                                  pool=self.pool4,
                                                                  batch=self.batch)
            feat3 = \
                self.tool2.roi_pooling_cuda(x3, proposal, label=None, stride=cfg.feature_stride, pool=self.pool3,
                                            batch=self.batch)[
                    0]
            feat2 = \
                self.tool2.roi_pooling_cuda(x2, proposal, label=None, stride=int(cfg.feature_stride / 2),
                                            pool=self.pool2,
                                            batch=self.batch)[0]
            feat1 = \
                self.tool2.roi_pooling_cuda(x1, proposal, label=None, stride=int(cfg.feature_stride / 4),
                                            pool=self.pool1,
                                            batch=self.batch)[0]

            if self.flag == 2:
                result = {}
                result['feat_8'] = feat1
                result['feat_16'] = feat2
                result['feat_32'] = feat3
                result['feat_64'] = feat4
                result['label'] = label
                result['predict_offset'] = 0
                result['class_num'] = class_num
                result['batch_cor_weight'] = 0
                result['cross_entropy'] = cross_entropy
                result['loss_box'] = loss_box
                return result
            elif self.flag == 3:
                result = {}
                result['feat_8'] = feat1
                result['feat_16'] = feat2
                result['feat_32'] = feat3
                result['feat_64'] = feat4
                result['label'] = label
                result['class_num'] = class_num
                result['cross_entropy'] = cross_entropy
                result['loss_box'] = loss_box

                return result
