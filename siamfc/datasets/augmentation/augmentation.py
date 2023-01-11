# Data augmentation for pcb dataset
import cv2
import ipdb
import numpy as np


class Augmentations(object):
    def __init__(self, clipLimit=3.0) -> None:
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit)

    def CLAHE(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        return bgr

    def horizontal_flip(self, img, box):
        img = img.copy()
        box = box.copy()

        img = cv2.flip(img, 1)  # 水平翻轉
        width = img.shape[1]
        bbox_x1 = width - box[:, 2] - 1  # x1 = w - x2
        bbox_x2 = width - box[:, 0] - 1  # x2 = w - x1
        box[:, 0] = bbox_x1
        box[:, 2] = bbox_x2
        return img, box

    def vertical_flip(self, img, box):
        img = img.copy()
        box = box.copy()

        img = cv2.flip(img, 0)  # 垂直翻轉
        img_h = img.shape[0]
        bbox_y1 = img_h - box[:, 3] - 1  # y1 = h - y2
        bbox_y2 = img_h - box[:, 1] - 1  # y2 = h - y1
        box[:, 1] = bbox_y1
        box[:, 3] = bbox_y2
        return img, box


class PCBAugmentation(object):
    def __init__(self, args) -> None:
        self.aug = Augmentations()
        self.clahe = args.clahe
        self.flip = args.flip

    def __call__(self, img, box):
        # Flip
        if self.flip and self.flip > np.random.random():
            img, box = self.aug.vertical_flip(img, box)
            img, box = self.aug.horizontal_flip(img, box)

        # CLAHE 3.0
        if self.clahe:
            img = self.aug.CLAHE(img)

        return img, box
