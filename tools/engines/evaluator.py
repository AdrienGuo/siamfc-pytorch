from typing import List

import ipdb
import numpy as np
import torch


class Evaluator(object):
    def __init__(self) -> None:
        pass

    def evaluate(self, model, dataloader):
        preds = list()
        labels = list()
        with torch.no_grad():
            for iter, data in enumerate(dataloader):
                img_name = data['img_name'][0]
                z_img = data['z_img']
                x_img = data['x_img']
                z_box = data['z_box'][0].numpy()
                gt_boxes = data['gt_boxes'][0].numpy()

                print(f"Load image: {img_name}")
                model.init(z_img, z_box)
                pred_boxes = model.match(x_img)

                preds.append(pred_boxes)
                labels.append(gt_boxes)

                # ipdb.set_trace()

        metrics = self.calculate_metrics(preds, labels)
        return metrics

    def calculate_metrics(self, preds: List[np.array], labels):
        tp = list()
        fp = list()
        boxes_num = list()

        for idx in range(len(preds)):
            # 一個 data
            tp_one = np.zeros(len(preds[idx]))
            fp_one = np.zeros(len(preds[idx]))
            for pred_idx in range(len(preds[idx])):
                # 這個 data 裡面的一個 pred_box
                best_iou = 0
                for label_idx in range(len(labels[idx])):
                    # 這個 data 裡面的一個 label_box
                    iou = Evaluator.overlap_ratio_one(
                        preds[idx][pred_idx], labels[idx][label_idx])
                    if iou > best_iou:
                        # 記錄這一個 pred_box 和所有 labels 最大的 iou
                        best_iou = iou
                # 根據 best_iou 判斷這一個 pred_box 是 TP 還是 FP
                # 所有預測出來的 pred_boxes，他們都已經是 positive 了；而且不是 true (TP) 就是 false (FP)
                if best_iou >= 0.5:
                    tp_one[pred_idx] = 1
                else:
                    fp_one[pred_idx] = 1

            tp_one_sum = sum(tp_one)
            fp_one_sum = sum(fp_one)
            boxes_one_num = len(labels[idx])  # 總共有多少個 gt

            tp.append(tp_one_sum)
            fp.append(fp_one_sum)
            boxes_num.append(boxes_one_num)

        # length of tp = len(preds)
        # length of fp = len(preds)
        # length of boxes_num = len(preds)
        if (sum(tp) + sum(fp) == 0):
            # 完全沒有預測出物件
            precision = 0.0
        else:
            precision = sum(tp) / (sum(tp) + sum(fp))
        recall = sum(tp) / sum(boxes_num)

        return {
            'Recall': recall * 100,
            'Precisioin': precision * 100,
        }

    def save_metrics(self, metrics, filepath):
        pass

    @staticmethod
    def overlap_ratio_one(rect1, rect2) -> float:
        """ Compute overlap ratio between two rects

        Args
            rect (np.array): (N, [x1, y1, x2, y2])
        Return:
            iou
        """

        left = np.maximum(rect1[0], rect2[0])
        right = np.minimum(rect1[2], rect2[2])
        top = np.maximum(rect1[1], rect2[1])
        bottom = np.minimum(rect1[3], rect2[3])
        rect1_size = np.array([rect1[2] - rect1[0], rect1[3] - rect1[1]])
        rect2_size = np.array([rect2[2] - rect2[0], rect2[3] - rect2[1]])

        intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
        union = np.prod(rect1_size) + np.prod(rect2_size) - intersect
        iou = intersect / union
        iou = np.maximum(np.minimum(1, iou), 0)
        return iou
