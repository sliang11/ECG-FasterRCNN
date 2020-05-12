from rpn_tool_b import rpn_tool_b
import torch
from config import cfg
# from box_utils import jaccard
from box_transform import box_to_offset, offset_to_box
from tool.roi_layers import nms

class rpn_tool_c(rpn_tool_b):

    def predict_select(self, confidence: torch.Tensor, box_predict: torch.Tensor, default_box: torch.Tensor,
                       filter=True):

        if len(confidence.size()) > 2:
            batch_proposal = []
            batch_conf = []
            batch_offset = []
            for index in range(confidence.size()[0]):

                # this_default_box = default_box[:confidence.size()[-2], :]
                this_default_box = default_box.clone()
                this_scores = confidence[index, :, 1]
                # this_scores = confidence[index, 1:].max()
                bbox_deltas = box_predict[index]

                this_proposal = offset_to_box(this_default_box, bbox_deltas)
                this_proposal = torch.clamp(this_proposal, min=cfg.left_border,
                                            max=cfg.right_border)
                keep = this_proposal[:, 0] <= this_proposal[:, 1]
                this_proposal = this_proposal[keep]
                bbox_deltas = bbox_deltas[keep]


                keep = this_scores >= 0.5
                this_proposal = this_proposal[keep]
                bbox_deltas = bbox_deltas[keep]
                this_scores = this_scores[keep]

                # ws = this_proposal[:, 1] - this_proposal[:, 0] + 1
                # min_keep = ws < cfg.box_max_size
                # max_keep = ws > cfg.box_min_size
                # keep = torch.min(min_keep, max_keep)
                # print('keep:{}'.format(keep.sum()))

                # this_proposal = this_proposals[keep]
                # this_scores = this_scores[keep]
                if this_proposal.size()[-2] > cfg.be_topk:

                    order = torch.argsort(this_scores)[-cfg.be_topk:]
                    this_proposal = this_proposal[order]
                    this_scores = this_scores[order]


                # keep2, count = nms2(this_proposal, this_scores)

                keep2 = nms(this_proposal, this_scores, cfg.nms_threash)
                keep2 = keep2[:cfg.af_topk]

                # print(count)
                batch_offset.append(bbox_deltas[keep2])
                this_proposal = this_proposal[keep2]
                batch_proposal.append(this_proposal)
                batch_conf.append(this_scores[keep2])
            return batch_proposal, batch_conf, batch_offset
        else:
            this_proposal = offset_to_box(default_box, box_predict)
            this_proposal = torch.clamp(this_proposal, min=cfg.left_border,
                                        max=cfg.right_border)
            keep = this_proposal[:, 0] < this_proposal[:, 1]

            this_proposal = this_proposal[keep]
            box_predict = box_predict[keep]
            confidence = confidence[:, 1]
            confidence = confidence[keep]
            #######################

            keep3 = confidence >= 0.5
            # keep3 = confidence >= 0.8
            if (keep3.sum().item() > 0):
                this_proposal = this_proposal[keep3]
                box_predict = box_predict[keep3]
                confidence = confidence[keep3]
            if this_proposal.size()[-2] > cfg.be_topk:

                order = torch.argsort(confidence)[-cfg.be_topk:]
                this_proposal = this_proposal[order]
                # confidence = confidence.view(-1, 1)[order]
                confidence = confidence[order]

            # keep2, count = nms2(this_proposal, confidence)

            keep2 = nms(this_proposal, confidence, cfg.nms_threash)
            keep2 = keep2[:cfg.af_topk]
            # print(count)
            this_proposal = this_proposal[keep2]
            this_conf = confidence[keep2]
            box_predict = box_predict[keep2]
            return this_proposal, this_conf, box_predict
