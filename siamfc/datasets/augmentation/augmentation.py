import cv2
import ipdb
import numpy as np


# Ref: https://github.com/albumentations-team/albumentations/blob/2a1826d49c9442ae28cf33ddef658c8e24505cf8/albumentations/augmentations/functional.py#L450
def clahe(img, clip_limit=3.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe_mat = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = clahe_mat.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    return img


class Augmentations(object):
    def __init__(self) -> None:
        pass

    def CLAHE(self, img):
        return clahe(img)

    def horizontal_flip(self, img, box=None):
        # img = img.copy()
        img = cv2.flip(img, 1)  # 水平翻轉
        width = img.shape[1]

        if box is not None:
            box = box.copy()
            box = np.stack([
                width - box[:, 2] - 1,
                box[:, 1],
                width - box[:, 0] - 1,
                box[:, 3]
            ], axis=1)

        return img, box

    def vertical_flip(self, img, box=None):
        # img = img.copy()
        img = cv2.flip(img, 0)  # 垂直翻轉
        img_h = img.shape[0]

        if box is not None:
            box = box.copy()
            bbox_y1 = img_h - box[:, 3] - 1  # y1 = h - y2
            bbox_y2 = img_h - box[:, 1] - 1  # y2 = h - y1
            box[:, 1] = bbox_y1
            box[:, 3] = bbox_y2

        return img, box
