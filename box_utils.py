# -*- coding: utf-8 -*-
import torch
import numpy as np
from config import cfg


def point_form(boxes):

    return torch.cat((boxes[:, :1] - boxes[:, 1:] / 2,
                      boxes[:, :1] + boxes[:, 1:] / 2), 1)


def center_size(boxes):

    return torch.cat((boxes[:, 1:] + boxes[:, :1]) / 2,  # cx, cy
                     boxes[:, 1:] - boxes[:, :1], 1)  # w, h


def intersect(box_a, box_b):

    A = box_a.size(0)
    B = box_b.size(0)


    max_xy = torch.min(box_a[:, 1:].unsqueeze(1).expand(A, B, 1),
                       box_b[:, 1:].unsqueeze(0).expand(A, B, 1))

    min_xy = torch.max(box_a[:, :1].unsqueeze(1).expand(A, B, 1),
                       box_b[:, :1].unsqueeze(0).expand(A, B, 1))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0]


def jaccard(box_a, box_b):

    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 1] - box_a[:, 0]).unsqueeze(1).expand_as(inter)
    area_b = (box_b[:, 1] - box_b[:, 0]).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    iou = inter / union
    nonoverlap = area_b - inter
    nonoverlap2 = area_a - inter
    return iou, union, nonoverlap,nonoverlap2


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    overlaps = jaccard(
        truths,
        priors
    )[0]
    maxlap_of_ground, maxidx_of_ground = overlaps.max(1, keepdim=True)


    maxlap_of_default, maxidx_of_default = overlaps.max(0, keepdim=True)

    maxidx_of_default.squeeze_(0)
    maxlap_of_default.squeeze_(0)
    maxidx_of_ground.squeeze_(1)
    maxlap_of_ground.squeeze_(1)

    maxlap_of_default.index_fill_(0, maxidx_of_ground, 2)
    for j in range(maxidx_of_ground.size(0)):
        maxidx_of_default[maxidx_of_ground[j]] = j

    matches = truths[maxidx_of_default]
    conf = labels[maxidx_of_default] + 1
    conf[maxlap_of_default < threshold] = 0
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf


def encode(matched, priors, variances):


    g_cxcy = (matched[:, :1] + matched[:, 1:]) / 2 - priors[:, :1]

    g_cxcy /= priors[:, 2:]


    g_wh = (matched[:, 1:] - matched[:, :1]) / priors[:, 1:]


    g_wh = torch.log(g_wh)  #

    return torch.cat([g_cxcy, g_wh], 1)


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):


    boxes = torch.cat((
        priors[:, :1] + loc[:, :1] * variances[0] * priors[:, 1:],
        priors[:, 1:] * torch.exp(loc[:, 1:] * variances[1])), 1)
    boxes[:, :1] -= boxes[:, 1:] / 2
    boxes[:, 1:] += boxes[:, :1]
    return boxes


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


def nms(boxes, scores, overlap=cfg.nms_threash, top_k=cfg.af_topk):
    pass