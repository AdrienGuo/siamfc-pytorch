import numpy as np
import torch


def create_labels(size, cfg):
    # skip if same sized labels already created
    # if hasattr(self, 'labels') and self.labels.size() == size:
    #     return self.labels

    def logistic_labels(x, y, r_pos, r_neg):
        dist = np.abs(x) + np.abs(y)  # block distance
        labels = np.where(dist <= r_pos,
                            np.ones_like(x),
                            np.where(dist < r_neg,
                                    np.ones_like(x) * 0.5,
                                    np.zeros_like(x)))
        return labels

    # distances along x- and y-axis
    n, c, h, w = size
    x = np.arange(w) - (w - 1) / 2
    y = np.arange(h) - (h - 1) / 2
    x, y = np.meshgrid(x, y)

    # create logistic labels
    r_pos = cfg.r_pos / cfg.total_stride
    r_neg = cfg.r_neg / cfg.total_stride
    labels = logistic_labels(x, y, r_pos, r_neg)

    # repeat to size
    labels = labels.reshape((1, 1, h, w))
    labels = np.tile(labels, (n, c, 1, 1))

    # convert to tensors
    labels = torch.from_numpy(labels).float()
    
    return labels
