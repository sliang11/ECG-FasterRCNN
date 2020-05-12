from rpn_tool_c import rpn_tool_c
import torch.nn.functional as F


class rpn_tool_d(rpn_tool_c):
    def get_proposal(self, predict_confidence, box_predict, gt_boxes, test=False):

        # gt_boxes = params['gt_box']
        # predict_confidence = params['predict_confidence']
        # box_predict = params['box_predict']
        # test = params['test']

        if test == False:
            default_label, default_gt_offset = self.default_gt_match(gt_boxes, self.default_box)
            cross_entropy, loss_box = self.build_loss(predict_confidence, box_predict, default_label,
                                                      default_gt_offset)
            return cross_entropy, loss_box
        else:
            predict_confidence = F.softmax(predict_confidence, -1)
            proposal, batch_conf, batch_offset = self.predict_select(predict_confidence, box_predict, self.default_box)
            return proposal, batch_conf, batch_offset
