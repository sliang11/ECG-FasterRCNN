import torch
import numpy as np


def box_to_offset(default_box, gt_box):
    widths = default_box[:, 1] - default_box[:, 0] + 1.0
    ctr_x = default_box[:, 0] + 0.5 * widths

    gt_widths = gt_box[:, 1] - gt_box[:, 0] + 1.0
    gt_ctr_x = gt_box[:, 0] + 0.5 * gt_widths

    targets_dx = ((gt_ctr_x - ctr_x) / widths)
    targets_dw = (torch.log(gt_widths / widths))

    targets = torch.stack([targets_dx, targets_dw], -1)
    return targets


def offset_to_box(default_box, offset):
    default_box = default_box.expand_as(offset)
    widths = default_box[:, 1] - default_box[:, 0] + 1.0
    ctr_x = default_box[:, 0] + 0.5 * widths


    dx = offset[:, 0]
    dw = offset[:, 1]

    pred_ctr_x = dx * widths + ctr_x
    pred_w = torch.exp(dw) * widths

    pred_default_box = torch.zeros(default_box.size()).cuda()
    # x1
    pred_default_box[:, 0] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_default_box[:, 1] = pred_ctr_x + 0.5 * pred_w


    return pred_default_box


def clip_predict_box(default_box, im_shape):
    if default_box.shape[0] == 0:
        return default_box
    # x1 >= 0
    default_box[:, 0::4] = np.maximum(np.minimum(default_box[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    default_box[:, 1::4] = np.maximum(np.minimum(default_box[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    default_box[:, 2::4] = np.maximum(np.minimum(default_box[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    default_box[:, 3::4] = np.maximum(np.minimum(default_box[:, 3::4], im_shape[0] - 1), 0)
    return default_box
