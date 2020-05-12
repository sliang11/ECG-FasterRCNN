import torch
import numpy as np


def mixup_data(x, y, alpha=2, use_cuda=True):

    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam, weight):
        return lambda criterion, pred: lam * criterion(pred, y_a, weight) + (1 - lam) * criterion(pred, y_b, weight)

