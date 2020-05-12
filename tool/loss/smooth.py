import torch
import torch.nn as nn



class smooth_focal_weight(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.002,
                 reduction='mean',
                 lb_ignore=255,
                 ):
        super(smooth_focal_weight, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label, weight):


        weight = [weight[i] for i in label]
        weight = torch.stack(weight).cuda()
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        # label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1),
                                                          1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1 - lb_one_hot)

        tmp = -torch.sum(logs * label, dim=1)
        tmp = tmp.pow(2)
        tmp = tmp * weight
        if self.reduction == 'mean':
            loss = torch.sum(tmp) / n_valid
        elif self.reduction == 'none':
            loss = tmp
        return loss
