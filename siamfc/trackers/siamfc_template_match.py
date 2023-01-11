from collections import namedtuple

import cv2
import ipdb
import numpy as np
import torch

from ..models.backbone.backbones import AlexNetV1
from ..models.head.heads import SiamFC
from ..models.model_builder import SiameseNet, SeperateNet


class SiamFCTemplateMatch():
    def __init__(self, model, **kwargs) -> None:
        self.net = model
        self.net.eval()
        self.cfg = self.parse_args(**kwargs)

        # Setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 1,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 1000,
            'batch_size': 8,
            'num_workers': 8,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    @torch.no_grad()
    def init(self, z_img, z_box):
        """
        用 template 做初始化 (z_img)，
        用 template 在 search image 的位置 (z_box) 做初始化。

        Args:
            z_img: (B=1, C, H, W)
            z_box: (N, 4=[x1, y1, x2, y2])
        """

        # Set to evaluation mode
        self.net.eval()

        z_img = z_img.to(self.device)
        # exemplar feature
        # self.z_f: (1, 256, 6, 6)
        self.z_f = self.net.backbone(z_img)
        # self.z_f = self.net.backbone_z(z_img)

        # 下面都跟 template 沒關係，是之後會用到的一些設定
        # 272 = 16 * 17
        # self.upscale_sz = self.cfg.response_up * self.cfg.response_sz

        # search scale factors
        # self.scale_factors = self.cfg.scale_step ** np.linspace(
        #     -(self.cfg.scale_num // 2),
        #     self.cfg.scale_num // 2, self.cfg.scale_num)

        if z_box.ndim > 1:
            z_box = z_box.squeeze()
        # z_box: [x1, y1, x2, y2] -> [cx, cy, w, h]
        z_box = np.array([
            (z_box[0] + z_box[2]) / 2,
            (z_box[1] + z_box[3]) / 2,
            (z_box[2] - z_box[0]),
            (z_box[3] - z_box[1]),
        ], dtype=np.float32)
        self.target_sz = z_box[2:]
        # exemplar and search sizes
        # context = 0.5 * (w+h)
        # context = self.cfg.context * np.sum(self.target_sz)
        # self.z_sz = ( (w + context) * (h + context) ) ^ (1/2)
        # self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        # self.x_sz = self.z_sz * \
        #     self.cfg.instance_sz / self.cfg.exemplar_sz

    @torch.no_grad()
    def match(self, x_img):
        """在 search image 上面做 matching。

        Args:
            x_img: (B, C, H, W)
        """

        # Set to evaluation mode
        self.net.eval()

        # 注意順序，是 (H, W)
        self.center = np.array([x_img.shape[-2] / 2, x_img.shape[-1] / 2])
        x_img = x_img.to(self.device)

        # search feature
        # x_f: (1, 256, 22, 22)
        x_f = self.net.backbone(x_img)
        # x_f = self.net.backbone_x(x_img)

        # response: (scale_num, 1, 17, 17)
        responses = self.net.head(self.z_f, x_f)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample response and penalize scale changes
        # response: (scale_num, 272, 272)
        responses = np.stack([
            cv2.resize(
                response,
                (response.shape[1], response.shape[0]),
                interpolation=cv2.INTER_CUBIC)
            for response in responses
        ])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        # scale_id: 在 scale_num 個裡面，最大的那個數 (選擇維度)
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        # response: (272, 272)
        response = responses[scale_id]
        response -= response.min()
        response /= response.max()
        response_size = response.shape[:2]

        boxes = list()
        threshold = 0.9
        locs = np.argwhere(response >= threshold)
        # Sort locs in descending order
        scores = response[locs[:, 0], locs[:, 1]]
        sort_idx = np.argsort(-scores)
        locs = locs[sort_idx]
        for loc in locs:
            # locate target center
            # 改成以 response 的中心點當作原點
            disp_x_in_response = loc[1] - ((response_size[1] - 1) / 2)
            disp_y_in_response = loc[0] - ((response_size[0] - 1) / 2)
            # disp_in_instance: 改換到 self.scale_sz 上的比例後 (除以 response_up)，
            #                   再乘上 total_stride 變成實際的位移
            disp_x_in_instance = disp_x_in_response * self.cfg.total_stride
            disp_y_in_instance = disp_y_in_response * self.cfg.total_stride
            center = self.center + (disp_y_in_instance, disp_x_in_instance)
            box = np.array([
                (center[1] + 1) - (self.target_sz[0] - 1) / 2,
                (center[0] + 1) - (self.target_sz[1] - 1) / 2,
                (center[1] + 1) + (self.target_sz[0] - 1) / 2,
                (center[0] + 1) + (self.target_sz[1] - 1) / 2
            ])
            boxes.append(box)
        # NMS
        boxes = np.array(boxes)
        overlapThresh = 0.1
        boxes = self.nms(boxes, overlapThresh)

        # # 注意： loc: [row, col] 是先 y 軸，再 x 軸
        # loc = np.unravel_index(response.argmax(), response.shape)

        # # locate target center
        # # 改成以 response 的中心點當作原點
        # disp_x_in_response = loc[1] - ((response_size[1] - 1) / 2)
        # disp_y_in_response = loc[0] - ((response_size[0] - 1) / 2)
        # # disp_in_instance: 改換到 self.scale_sz 上的比例後 (除以 response_up)，
        # #                   再乘上 total_stride 變成實際的位移
        # disp_x_in_instance = disp_x_in_response * self.cfg.total_stride
        # disp_y_in_instance = disp_y_in_response * self.cfg.total_stride
        # self.center += (disp_y_in_instance, disp_x_in_instance)

        # # 這裡要超級超級超級小心上面 (x,y) 的順序
        # # box: [x1, y1, x2, y2]
        # box = np.array([
        #     (self.center[1] + 1) - (self.target_sz[0] - 1) / 2,
        #     (self.center[0] + 1) - (self.target_sz[1] - 1) / 2,
        #     (self.center[1] + 1) + (self.target_sz[0] - 1) / 2,
        #     (self.center[0] + 1) + (self.target_sz[1] - 1) / 2])
        # boxes = np.expand_dims(box, axis=0)

        # ipdb.set_trace()

        return boxes

    def nms(self, boxes, overlapThresh=0.1):
        # Return an empty list, if no boxes given
        if len(boxes) == 0:
            return []
        x1 = boxes[:, 0]  # x coordinate of the top-left corner
        y1 = boxes[:, 1]  # y coordinate of the top-left corner
        x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
        y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
        # Compute the area of the bounding boxes and sort the bounding
        # Boxes by the bottom-right y-coordinate of the bounding box
        areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts

        pick = list()
        # The indices of all boxes at start. We will redundant indices one by one.
        indices = np.arange(len(x1))
        while len(indices) > 0:
            idx = indices[0]
            pick.append(idx)

            # Find out the coordinates of the intersection box
            xx1 = np.maximum(boxes[idx, 0], boxes[indices[1: ], 0])
            yy1 = np.maximum(boxes[idx, 1], boxes[indices[1: ], 1])
            xx2 = np.minimum(boxes[idx, 2], boxes[indices[1: ], 2])
            yy2 = np.minimum(boxes[idx, 3], boxes[indices[1: ], 3])
            # Find out the width and the height of the intersection box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            # Compute ious
            overlaps = (w * h)
            ious = overlaps / (areas[idx] + areas[indices[1: ]] - overlaps)
            # Find 小於 threshold 的 ious
            idx = np.where(ious <= overlapThresh)[0]
            indices = indices[idx + 1]  # 留下那些框

        return boxes[pick]