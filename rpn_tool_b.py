from rpn_tool_a import rpn_tool_a
import torch
from config import cfg
from box_utils import nms, jaccard
from box_transform import box_to_offset, offset_to_box


class rpn_tool_b(rpn_tool_a):

    def default_gt_match(self, gt_box: [torch.Tensor], default_box: torch.Tensor):

        default_box = torch.clamp(default_box, min=0,
                                  max=cfg.right_border)

        batch_offset = []
        batch_label = []

        for index in range(len(gt_box)):

            gt_box_x = gt_box[index]
            # tmp = gt_box_x[:, -1]
            keep = gt_box_x[:, -1] >= 0
            gt_box_x = gt_box_x[keep]
            # gt_label = gt_box_x[:, -1]
            default_label = torch.Tensor(default_box.size()[0]).fill_(-1)
            # default_label = torch.Tensor(default_box.size()[0]).fill_(0)

            gt_box_x = gt_box_x[:, :2].cuda()


            overlap, union, non_overlap,tt = jaccard(gt_box_x, default_box)

            maxlap_of_ground, maxidx_of_ground = overlap.max(1)

            maxlap_of_default, maxidx_of_default = overlap.max(0)

            nonlap_of_default, nonidx_of_default = non_overlap.max(0)

            maxlap_of_default.index_fill_(0, maxidx_of_ground, 2)

            if len(maxidx_of_ground.size()) >= 1:
                for j in range(maxidx_of_ground.size()[0]):
                    # maxlap_of_default[maxidx_of_ground[j]] = maxlap_of_ground[j]
                    maxidx_of_default[maxidx_of_ground[j]] = j
            else:
                maxidx_of_default[maxidx_of_ground] = 0

            # tmp = nonlap_of_default < 39
            # tmp2 = nonlap_of_default >= 39
            # matches = gt_box_x[nonidx_of_default].squeeze()
            ################################
            tmp = maxlap_of_default < cfg.rpn_neg_thresh
            tmp2 = maxlap_of_default >= cfg.rpn_pos_thresh
            matches = gt_box_x[maxidx_of_default].squeeze()
            ################################
            default_label[tmp] = 0
            default_label[tmp2] = 1

            default_gt_offset = box_to_offset(default_box, matches)
            # t=default_gt_offset[default_label>0]
            batch_offset.append(default_gt_offset)
            batch_label.append(default_label.cuda())

        return batch_label, batch_offset
