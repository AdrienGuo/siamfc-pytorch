import cv2
import numpy as np


def draw_boxes(img, boxes, type=None):
    """
    Args:
        boxes: (N, 4=[x1, y1, x2, y2])
    """

    # Layout of the output array img is incompatible with cv::Mat
    img = img.copy()
    boxes = boxes.copy()
    boxes = np.array(boxes, dtype=np.int32)
    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, axis=0)

    if type == "template":
        color = (0, 0, 255)  # red
        thickness = 2
    elif type == "pred":
        color = (0, 255, 0)  # green
        thickness = 2
    elif type == "gt":
        color = (255, 0, 0)  # blue
        thickness = 3
    else:
        color = (255, 255, 255)  # white?
        thickness = 1

    for i, box in enumerate(boxes):
        pt1 = (box[0], box[1])
        pt2 = (box[2], box[3])
        img = cv2.rectangle(img, pt1, pt2, color=color, thickness=thickness)

    return img
