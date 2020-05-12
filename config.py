from easydict import EasyDict as edict
from math import ceil
import torch

cfg = edict()
cfg.mb = edict()
# cfg.output_stride = 128
cfg.right_border = 4544
cfg.left_border = 64
# cfg.roi_ratio = 0.5
cfg.anchor_scale = [2, 3, 4]
cfg.anchor_ratio = [2, 3, 4]
cfg.test = False
cfg.feature_stride = 64
cfg.feature_length = 71
cfg.base_size = 64

cfg.num_box = len(cfg.anchor_scale)
cfg.rpn_neg_thresh = 0.3
cfg.rpn_pos_thresh = 0.7
cfg.nms_threash = 0.3
cfg.classes = 5
cfg.roi_neg_thresh = 0.3
cfg.roi_pos_thresh = 0.7

cfg.testing_metrics = '150ms'
cfg.test_threashold = 54
cfg.noise_scale = 0.05
# cfg.noise_offset = 1
cfg.be_topk = 100
cfg.af_topk = 30

cfg.mb_ori_ltdb_set = [
    [2, 64, 2, 2],
    [3, 128, 2, 2],
    [3, 256, 3, 2],
    [3, 512, 3, 2],
    [2, 1024, 2, 2]]

##############################
# cfg.feature_mode = 'all'
# cfg.pool_fusion = False
# cfg.save_data = False
# cfg.save_dict = {}

