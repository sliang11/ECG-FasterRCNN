from rcn_tool_a import rcn_tool_a
import torch
from config import cfg
class rcn_tool_b(rcn_tool_a):
    def roi_pooling_cuda(self, features, proposal, label=None, stride=cfg.feature_stride, pool=None, batch=False):
        if batch == True:
            batch_output = []
            batch_label = []
            if label != None:
                batch_label.extend([j for i in label for j in i])
                batch_label = torch.stack(batch_label)
            outputs = pool(features, proposal)
            batch_output = outputs
            class_num = [0] * 6
            # if label != None:
            #     for i in batch_label:
            #         if i != -1:
            #             class_num[i.item()] += 1
            #     average = int(sum(class_num) / 6)
            #     class_num = [average / (i + 1) for i in class_num]
            return batch_output, batch_label, class_num
        else:
            if len(features.size()) == 3:
                batch_size, num_channels, data_width = features.size()
                batch_output = []
                batch_label = []
                for index in range(batch_size):
                    data = features[index]
                    this_proposal = proposal[index]
                    # num_proposal = this_proposal.size()[0]
                    outputs = pool(data, this_proposal)
                    # if torch.isnan(outputs).sum()>=1:
                    #     print('nan produce')
                    # if torch.isinf(outputs).sum()>=1:
                    #     print('inf procude')
                    batch_output.append(outputs)
                    if label != None:
                        batch_label.extend([i for i in label[index]])
                if label != None:
                    batch_label = torch.stack(batch_label)
                # batch_output = [torch.stack(i) for i in batch_output]

                class_num = [0] * 5
                # if label != None:
                #     for i in batch_label:
                #         if i != -1:
                #             class_num[i.item()] += 1
                #     average = int(sum(class_num) / 5)
                #     class_num = [average / (i + 1) for i in class_num]
                # class_num[0] /= 30
                return batch_output, batch_label, class_num
            else:
                batch_output = []
                batch_label = []
                # num_channels, data_width = features.size()
                data = features
                this_proposal = proposal
                num_proposal = this_proposal.size()[0]
                # width_limit_right = torch.Tensor([data_width - 1] * num_proposal).cuda()
                # width_limit_left = torch.zeros(num_proposal).cuda()
                # start = torch.floor(this_proposal * (1 / stride))[:,
                #         0]
                # end = torch.ceil(this_proposal * (1 / stride))[:, 1]  #
                # wstart = torch.min(width_limit_right, torch.max(width_limit_left, start)).type(
                #     torch.long)
                # wend = torch.min(width_limit_right, torch.max(width_limit_left, end)).type(
                #     torch.long)
                # tmp = self.get_average(data, wstart, wend, stride)
                outputs = pool(data, this_proposal)
                # outputs = tmp
                batch_output.extend([outputs[i, :] for i in range(num_proposal)])
                if label != None:
                    batch_label.extend(label)
                batch_output = torch.stack(batch_output, 0)
                return batch_output