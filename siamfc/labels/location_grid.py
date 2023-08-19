import ipdb
import torch
from config.config import cfg


def create_locations(map_size, img_size):
    h, w = map_size[-2:]
    img_h, img_w = img_size[-2:]
    locations = compute_locations(h, w, img_h, img_w)
    return locations


def compute_locations(h, w, img_h, img_w):
    grids_x = torch.arange(
        0, w * cfg.total_stride, step=cfg.total_stride,
        dtype=torch.float32
    )
    grids_y = torch.arange(
        0, h * cfg.total_stride, step=cfg.total_stride,
        dtype=torch.float32
    )
    grids_y, grids_x = torch.meshgrid((grids_y, grids_x))
    # width & height are independent
    # 位移 = (原圖大小 - (score大小-1)*8) // 2
    shift_x = (img_w - (w - 1) * 8) / 2  # x 軸的起始點
    shift_y = (img_h - (h - 1) * 8) / 2  # y 軸的起始點
    grids_x = grids_x.reshape(-1) + shift_x
    grids_y = grids_y.reshape(-1) + shift_y
    locations = torch.stack((grids_x, grids_y), dim=1)    # alex:48 // 32
    return locations
