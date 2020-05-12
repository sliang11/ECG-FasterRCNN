import torch
import numpy as np
from rpn_initor import rpn_initor
from config import cfg


class eval_tool(rpn_initor):
    def first_process(self, info: dict, params: dict):
        gt_label = params.get('gt_label')
        maxlap_of_ground = params.get('maxlap_of_ground')
        maxidx_of_ground = params.get('maxidx_of_ground')
        nonlap_of_pre = params.get('minlap_of_pre').type(torch.LongTensor)
        nonlap_idx_of_pre = params.get('minidx_of_pre').type(torch.LongTensor)

        nonlap_of_ground = params.get('minlap_of_ground').type(torch.LongTensor)
        nonlap_idx_of_ground = params.get('minidx_of_ground').type(torch.LongTensor)

        pre_window = params.get('pre_window')
        ground_window = params['ground_window']
        peak = params['peak']
        num = params['num']
        tmp = {}
        tmp.setdefault('tp', 0)
        tmp.setdefault('fp', 0)
        tmp.setdefault('tn', 0)
        tmp.setdefault('fn', 0)
        tmp.setdefault('pre', [])
        tmp.setdefault('gt', [])
        tmp.setdefault('pre_bin', [])
        tmp.setdefault('gt_bin', [])
        view = torch.zeros(len(gt_label))
        save_dict = params['save']
        save_dict2 = {}
        fp_window = []
        fp_score = []

        fn_window = []
        fn_score = []

        mapping = torch.zeros(len(pre_window))
        for i in range(len(pre_window)):
            start = pre_window[i][0].item()
            end = pre_window[i][1].item()
            count = 0
            for j in range(len(peak)):
                if start <= peak[j] and end >= peak[j]:
                    count += 1
            if count == 1:
                mapping[i] = 1


        threshold = cfg.test_threashold



        # for i in range(len(nonlap_idx_of_pre)):
        #     i_ = int(nonlap_idx_of_pre[i].item())
        #     if nonlap_of_pre[i] <= 54 or nonlap_of_ground[i_] <= 54:
        #         view[i_] = 1

        for i in range(len(nonlap_idx_of_pre)):
            i_ = int(nonlap_idx_of_pre[i].item())
            view[i_] = 1
        total_1 = nonlap_of_pre <= threshold
        total_2 = mapping == 1
        keep = total_1 * total_2

        tmp.get('pre_bin').extend([1] * keep.sum().item())
        tmp.get('gt_bin').extend([1] * keep.sum().item())
        tmp['tp'] += keep.sum().item()

        # fp
        total_1 = nonlap_of_pre > threshold
        total_2 = mapping == 0


        keep = total_1 * total_2
        # tmp.get('gt').extend(label.tolist())
        # tmp.get('pre').extend(label.tolist())
        tmp.get('pre_bin').extend([1] * keep.sum().item())
        tmp.get('gt_bin').extend([0] * keep.sum().item())
        tmp['fp'] += keep.sum().item()
        fp_window.extend(pre_window[keep.cpu()].numpy())
        fp_score.extend(nonlap_of_pre[keep.cpu()].cpu().numpy())

        res = (view == 0)
        # res = torch.Tensor(res).type(torch.uint8)
        # label = gt_label[res]
        # tmp.get('gt').extend(label.tolist())
        # tmp.get('pre').extend([0] * res.sum().item())
        tmp.get('pre_bin').extend([0] * res.sum().item())
        tmp.get('gt_bin').extend([1] * res.sum().item())
        fn_window.extend(ground_window[res])
        tmp['fn'] += res.sum().item()

        # info.get('pre').extend(tmp.get('pre'))
        # info.get('gt').extend(tmp.get('gt'))
        info.get('pre_bin').extend(tmp.get('pre_bin'))
        info.get('gt_bin').extend(tmp.get('gt_bin'))
        info['tp'] += tmp.get('tp')
        info['fp'] += tmp.get('fp')
        info['fn'] += tmp.get('fn')

