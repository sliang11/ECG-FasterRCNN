import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha).cuda()
        self.size_average = size_average

    def forward(self, pre, ground):
        if pre.dim() > 2:
            pre = pre.view(pre.size(0), pre.size(1), -1)  # N,C,H,W => N,C,H*W
            pre = pre.transpose(1, 2)  # N,C,H*W => N,H*W,C
            pre = pre.contiguous().view(-1, pre.size(2))  # N,H*W,C => N*H*W,C
        ground = ground.view(-1, 1)

        logpt = F.log_softmax(pre, dim=-1)
        logpt = logpt.gather(1, ground)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != pre.data.type():
                self.alpha = self.alpha.type_as(pre.data)
            at = self.alpha.gather(0, ground.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


