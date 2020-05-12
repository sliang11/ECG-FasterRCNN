from rcn_tool_b import rcn_tool_b
import torch
class rcn_tool_c(rcn_tool_b):
    def cal_loss2(self, cls_score, label):

        cross_entropy = 0
        loss_box = 0
        # predict_offset = [j for i in predict_offset for j in i]
        # predict_offset = torch.stack(predict_offset, 0)

        # cls_weight = torch.FloatTensor(cls_weight)
        keep = label != -1
        # cor_weight = [j for i in cor_weight for j in i]
        # cor_weight = torch.stack(cor_weight).cuda()
        # cor_weight = cor_weight[keep]
        this_cls_score = cls_score[keep]
        this_label = label[keep]
        # cross_entropy += F.cross_entropy(this_cls_score, this_label.cuda(), weight=cls_weight.cuda(),
        #                                  ignore_index=5, reduction='sum')
        cross_entropy += self.loss(this_cls_score, this_label.cuda())
        cross_entropy = cross_entropy / len(keep)
        return cross_entropy

    def cal_loss(self, cls_score, box_pred, label, predict_offset, cls_weight, cor_weight):

        cross_entropy = 0
        loss_box = 0
        # predict_offset = [j for i in predict_offset for j in i]
        # predict_offset = torch.stack(predict_offset, 0)

        # cls_weight = torch.FloatTensor(cls_weight)
        keep = label != -1
        # cor_weight = [j for i in cor_weight for j in i]
        # cor_weight = torch.stack(cor_weight).cuda()
        # cor_weight = cor_weight[keep]
        this_cls_score = cls_score[keep]
        this_label = label[keep]
        # cross_entropy += F.cross_entropy(this_cls_score, this_label.cuda(), weight=cls_weight.cuda(),
        #                                  ignore_index=5, reduction='sum')
        cross_entropy += self.loss(this_cls_score, this_label.cuda())
        cross_entropy = cross_entropy / len(keep)
        return cross_entropy