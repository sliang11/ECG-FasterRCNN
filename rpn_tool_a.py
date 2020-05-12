from rpn_tool_init import rpn_initor
import torch.nn.functional as F
import torch

class rpn_tool_a(rpn_initor):
    def build_loss(self, predict_confidence, box_predict, default_label, default_gt_offset):
        # classification loss
        rpn_cross_entropy = 0
        rpn_loss_box = 0
        for index in range(len(default_label)):
            this_default_label = default_label[index]
            this_predict_confidence = predict_confidence[index]
            keep = this_default_label >= 0
            this_predict_confidence = this_predict_confidence[keep]
            this_default_label = this_default_label[keep].type(torch.long)
            rpn_cross_entropy += self.loss(this_predict_confidence, this_default_label)

            pos_index = default_label[index] > 0

            this_box_predict = box_predict[index]
            this_default_gt_offset = default_gt_offset[index]
            a = this_box_predict[pos_index]
            b = this_default_gt_offset[pos_index]

            rpn_loss_box += F.smooth_l1_loss(a, b, reduction='sum') * 10

        rpn_loss_box = rpn_loss_box / len(default_label)
        return rpn_cross_entropy, rpn_loss_box
