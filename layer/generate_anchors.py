import numpy as np
from config import cfg
import torch


class gene_default():
    def __init__(self):
        pass
    def generate_anchors(self):

        self.base_anchor = np.array([1, cfg.base_size]) - 1
        anchor = self._ratio_enum()
        return torch.Tensor(anchor)

    def phy_to_centroid(self, anchor):


        w = anchor[1] - anchor[0] + 1
        x_ctr = anchor[0] + 0.5 * (w - 1)
        return w, x_ctr

    def centroid_to_phy(self, ws, x_ctr):


        x_ctr=0
        anchor=[x_ctr-0.5*(ws-1),x_ctr+0.5*(ws-1)]
        anchor=np.stack(anchor,1)

        return anchor

    def _ratio_enum(self):

        w, x_ctr = self.phy_to_centroid(self.base_anchor)
        w = [w * i for i in cfg.anchor_ratio]
        w = np.asanyarray(w)
        anchors = self.centroid_to_phy(w, x_ctr)
        return anchors

    def _scale_enum(self, anchor, scales):


        w, x_ctr = self.phy_to_centroid(anchor)
        ws = w * scales
        anchors = self.centroid_to_phy(ws, x_ctr)
        return anchors


if __name__ == '__main__':
    pass
