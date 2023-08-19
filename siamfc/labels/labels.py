import ipdb
import numpy as np
import torch

from config.config import cfg

from .location_grid import create_locations


class Label(object):
    def __init__(self) -> None:
        # Setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

    def create_labels(self, map_size, img_size, gt_boxes):
        """
        Args:
            map_size: (B, C, H, W)
            img_size: (B, C, H, W)
            gt_boxes: (B, N, 4)

        Return:
            labels: (B, 1, H, W)
        """
        locations = create_locations(map_size, img_size)
        locations = locations.to(self.device)
        # labels = self.compute_labels_for_locations(
        #     locations, gt_boxes, map_size)
        labels = self.car_compute_labels_for_locations(
            locations, gt_boxes, map_size)

        return labels.to(self.device)

    def car_compute_labels_for_locations(self, locations, gt_boxes, map_size):
        # xs: (w*h)
        xs, ys = locations[:, 0], locations[:, 1]
        batch = map_size[0]
        h, w = map_size[2], map_size[3]

        bboxes = gt_boxes.squeeze(1)
        labels = torch.zeros(h*w, batch)

        # xs[:, None]: (w*h, 1)
        # bboxes[:, 0][None]: (1, B)
        # l: (w*h, B)
        l = xs[:, None] - bboxes[:, 0][None].float()
        t = ys[:, None] - bboxes[:, 1][None].float()
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        # reg_targets_per_im: (w*h, B, 4)
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

        s1 = reg_targets_per_im[:, :, 0] > 0.6 * \
            ((bboxes[:, 2]-bboxes[:, 0])/2).float()
        s2 = reg_targets_per_im[:, :, 2] > 0.6 * \
            ((bboxes[:, 2]-bboxes[:, 0])/2).float()
        s3 = reg_targets_per_im[:, :, 1] > 0.6 * \
            ((bboxes[:, 3]-bboxes[:, 1])/2).float()
        s4 = reg_targets_per_im[:, :, 3] > 0.6 * \
            ((bboxes[:, 3]-bboxes[:, 1])/2).float()
        is_in_boxes = s1*s2*s3*s4
        pos = np.where(is_in_boxes.cpu() == 1)
        labels[pos] = 1

        labels = labels.permute(1, 0).contiguous()
        labels = labels[:, None, :]
        labels = labels.view(batch, 1, h, w)

        return labels.contiguous()

    def compute_labels_for_locations(self, locations, gt_boxes, map_size):
        """有在 box 裡面的設定為 1，其他為 0
        Args:
            locations: (H x W, 2)
            gt_boxes: (B, N, 4)
            map_size: (B, C, H, W)

        Return:
            labels: (B, 1, H, W)
        """

        xs, ys = locations[:, 0], locations[:, 1]
        batch_size = map_size[0]
        map_h, map_w = map_size[2], map_size[3]

        gt_cls_all = list()

        for i in range(batch_size):
            match_boxes = gt_boxes[i]
            l = xs[:, None] - match_boxes[:, 0][None].float()
            t = ys[:, None] - match_boxes[:, 1][None].float()
            r = match_boxes[:, 2][None].float() - xs[:, None]
            b = match_boxes[:, 3][None].float() - ys[:, None]
            # N 為所有採樣點的數，G 為 match_boxes 數量
            # reg_targets_per_im: (N, G, 4)
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            # 整個 match_boxes 的區域都是 in_boxes
            # is_in_boxes: (N, G)
            # is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            # 不要整個 match_boxes 的區域都是 in_boxes
            # 只留下最中心的 0.3 x 0.3 的區塊
            # 可是要小心會不會變成完全沒有 in_boxes，因為我有些物件很小
            # RuntimeError: Integer division of tensors using div or / is no longer supported,
            # and in a future release div will perform true division as in Python 3.
            # Use true_divide or floor_divide (// in Python) instead
            s1 = reg_targets_per_im[:, :, 0] > \
                0.6 * ((match_boxes[:, 2] - match_boxes[:, 0]) / 2.).float()
            s2 = reg_targets_per_im[:, :, 2] > \
                0.6 * ((match_boxes[:, 2] - match_boxes[:, 0]) / 2.).float()
            s3 = reg_targets_per_im[:, :, 1] > \
                0.6 * ((match_boxes[:, 3] - match_boxes[:, 1]) / 2.).float()
            s4 = reg_targets_per_im[:, :, 3] > \
                0.6 * ((match_boxes[:, 3] - match_boxes[:, 1]) / 2.).float()
            # is_in_boxes: (N, G)
            is_in_boxes = s1 * s2 * s3 * s4

            # is_in_boxes: (G, N)
            is_in_boxes = is_in_boxes.permute(1, 0).contiguous()
            # gt_cls_per_im: (G, size * size)
            gt_cls_per_im = torch.zeros(
                is_in_boxes.shape[0], map_h * map_w)
            for g in range(is_in_boxes.shape[0]):
                pos = np.where(is_in_boxes[g].cpu() == 1)
                gt_cls_per_im[g][pos] = 1
            gt_cls_all.append(gt_cls_per_im)

        gt_cls_all = torch.stack(gt_cls_all, axis=0)
        return np.array(gt_cls_all)


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
