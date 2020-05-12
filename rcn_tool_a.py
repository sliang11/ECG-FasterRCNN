from rcn_tool_init import rcn_tool_init
import torch
from config import cfg

from box_utils import jaccard
from tool.roi_layers import nms
# from tool.box_transform import offset_to_box
from box_transform import offset_to_box, clip_predict_box, box_to_offset


class rcn_tool_a(rcn_tool_init):
    def predict_gt_match(self, proposal: list, gt_box: list, flag=0):
        batch_proposal = []
        batch_label = []
        batch_predict_offset = []
        batch_weight = []
        batch_pre_weight = []
        if flag == 0:
            for index in range(len(proposal)):
                keep = gt_box[index][:, 2] != -1
                this_gt_label = gt_box[index][keep][:, 2]
                this_proposal = proposal[index]
                this_gt_box = gt_box[index][keep][:, :2].cuda()
                this_noised_gt_box = self.gt_box_add_noise(this_gt_box)

                this_proposal = torch.cat([this_proposal, this_noised_gt_box, this_gt_box], 0)
                this_proposal = torch.clamp(this_proposal, min=cfg.left_border,
                                            max=cfg.right_border)

                overlap = jaccard(this_gt_box, this_proposal)[0]
                maxlap_of_ground, maxidx_of_ground = overlap.max(1, keepdim=True)

                maxlap_of_predict, maxidx_of_predict = overlap.max(0, keepdim=True)
                maxlap_of_ground = maxlap_of_ground.squeeze()
                maxidx_of_ground = maxidx_of_ground.squeeze()
                maxlap_of_predict = maxlap_of_predict.squeeze()
                maxidx_of_predict = maxidx_of_predict.squeeze()

                this_matches = this_gt_box[maxidx_of_predict]
                this_pre_label = this_gt_label[maxidx_of_predict]
                b = maxlap_of_predict < cfg.roi_neg_thresh
                this_pre_label[b] = 0

                each_label_num = [0] * 5
                # for i in range(5):
                #     each_label_num[i] = torch.sum(this_pre_label == i).item()

                keep = this_pre_label != -1
                batch_weight.append(torch.Tensor([1 for i in each_label_num]))
                this_proposal = this_proposal[keep]
                this_pre_label = this_pre_label[keep].type(torch.long)
                this_offset = box_to_offset(this_proposal, this_matches[keep])

                # this_predict_offset = regression_label(this_offset, this_pre_label)

                # this_predict_offset = this_predict_offset.reshape(-1, para.classes * 2)
                this_pre_weight = [(max(each_label_num) + 1) / (i + 1) for i in each_label_num]
                this_pre_weight = [this_pre_weight[i] for i in this_pre_label]
                this_pre_weight = torch.Tensor(this_pre_weight).view(-1, 1)
                this_pre_weight = torch.cat([this_pre_weight, this_pre_weight], 1)

                batch_label.append(this_pre_label.cuda())
                batch_proposal.append(this_proposal)
                batch_predict_offset.append(this_offset)
                batch_pre_weight.append(this_pre_weight)

            return batch_proposal, batch_label, batch_predict_offset, batch_weight, batch_pre_weight
        else:
            keep = gt_box[:, 2] != -1
            this_gt_label = gt_box[keep][:, 2]
            this_proposal = proposal
            this_gt_box = gt_box[keep][:, :2].cuda()

            this_proposal = torch.clamp(this_proposal, min=cfg.left_border,
                                        max=cfg.right_border)
            # keep = this_proposal[:, 0] <= this_proposal[:, 1]
            # this_proposal = this_proposal[keep]

            overlap, union, nonoverlap = jaccard(this_gt_box, this_proposal)
            maxlap_of_ground, maxidx_of_ground = overlap.max(1)

            maxlap_of_predict, maxidx_of_predict = overlap.max(0)
            nonlap_of_predict, nonlapidx_of_predict = nonoverlap.min(0)
            #
            this_matches = this_gt_box[maxidx_of_predict]
            this_pre_label = this_gt_label[maxidx_of_predict]

            # a = maxlap_of_predict > cfg.roi_neg_thresh_low
            if cfg.testing_metrics == '80':
                b = maxlap_of_predict <= 0.82
                this_pre_label[b] = 0  #
            elif cfg.testing_metrics == '150ms':
                # this_matches = this_gt_box[nonlapidx_of_predict]
                this_pre_label = this_gt_label[nonlapidx_of_predict]
                b = nonlap_of_predict >= 54
                this_pre_label[b] = 0

        return this_pre_label

    def gt_box_add_noise(self, gt_box, noise=cfg.noise_scale):

        noise_box = gt_box.clone().cuda()
        ws = noise_box[:, 1] - noise_box[:, 0] + 1.0

        tmp = torch.rand(noise_box.shape[0]).cuda()
        width_offset = tmp * noise * ws
        noise_box[:, 0] += width_offset
        noise_box[:, 1] += width_offset
        return noise_box

    def pre_gt_match_uniform(self, proposal, gt_box, training=True, params=None):

        batch_proposal = []
        batch_label = []
        if training == True:
            for index in range(len(proposal)):
                peak = params['peak'][index]
                keep = gt_box[index][:, 2] != -1
                this_gt_label = torch.LongTensor(gt_box[index][keep][:, 2].tolist())
                this_proposal = proposal[index]
                this_gt_box = gt_box[index][keep][:, :2].cuda()
                this_noised_gt_box = self.gt_box_add_noise(this_gt_box)

                this_proposal = torch.cat([this_proposal, this_noised_gt_box, this_gt_box], 0)
                this_proposal = torch.clamp(this_proposal, min=cfg.left_border,
                                            max=cfg.right_border)

                overlap, union, nonoverlap, tt = jaccard(this_gt_box, this_proposal)

                nonlap_of_predict, nonlapidx_of_predict = nonoverlap.min(0)
                maxlap_of_predict, maxidx_of_predict = overlap.max(0)
                # maxlap_of_predict = maxlap_of_predict.type(torch.long)
                nonlap_of_predict = nonlap_of_predict.type(torch.long)
                # maxlap_of_ground, maxidx_of_ground = overlap.max(1)
                # this_matches = this_gt_box[maxidx_of_predict]
                this_pre_label = this_gt_label[maxidx_of_predict]
                # this_pre_label = torch.zeros(len(maxidx_of_predict))

                if 1:
                    mapping = torch.zeros(len(this_pre_label)).cuda()
                    for i in range(len(this_pre_label)):
                        start = this_proposal[i][0].item()
                        end = this_proposal[i][1].item()
                        count = 0
                        for j in range(len(peak)):
                            if start <= peak[j] and end >= peak[j]:
                                count += 1
                        if count == 1:
                            mapping[i] = 1
                        elif count > 1:
                            mapping[i] = 2

                    total_1 = nonlap_of_predict <= 54
                    total_2 = mapping == 1
                    keep = total_1 * total_2
                    for i in range(len(keep)):
                        if keep[i] == 1:
                            pass
                            # item = nonlapidx_of_predict[i]
                            # this_pre_label[i] = this_gt_label[item]
                        else:
                            this_pre_label[i] = 0

                this_pre_label = this_pre_label.type(torch.long)
                batch_label.append(this_pre_label.cuda())
                batch_proposal.append(this_proposal)
            return batch_proposal, batch_label
        else:
            peak = params['peak']
            keep = gt_box[:, 2] != -1
            this_gt_label = gt_box[keep][:, 2]
            this_proposal = proposal
            this_gt_box = gt_box[keep][:, :2].cuda()

            this_proposal = torch.clamp(this_proposal, min=cfg.left_border,
                                        max=cfg.right_border)

            overlap, union, nonoverlap, tt = jaccard(this_gt_box, this_proposal)
            maxlap_of_ground, maxidx_of_ground = overlap.max(1)

            maxlap_of_predict, maxidx_of_predict = overlap.max(0)
            nonlap_of_predict, nonlapidx_of_predict = nonoverlap.min(0)
            maxlap_of_predict = maxlap_of_predict.type(torch.long)
            nonlap_of_predict = nonlap_of_predict.type(torch.long)
            this_matches = this_gt_box[maxidx_of_predict]
            this_pre_label = this_gt_label[maxidx_of_predict]
            # this_pre_label = torch.zeros(len(maxidx_of_predict)).cuda()

            mapping = torch.zeros(len(this_pre_label)).cuda()
            for i in range(len(this_pre_label)):
                start = this_proposal[i][0].item()
                end = this_proposal[i][1].item()
                count = 0
                for j in range(len(peak)):
                    if start <= peak[j] and end >= peak[j]:
                        count += 1
                if count == 1:
                    mapping[i] = 1
                elif count > 1:
                    mapping[i] = 2

            total_1 = nonlap_of_predict <= 54
            total_2 = mapping == 1
            keep = total_1 * total_2
            for i in range(len(keep)):
                if keep[i] == 1:
                    pass
                    # item = nonlapidx_of_predict[i]
                    # this_pre_label[i] = this_gt_label[item]
                else:
                    this_pre_label[i] = 0
            this_pre_label = this_pre_label.type(torch.long)

        return this_pre_label
