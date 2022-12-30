import ipdb
import numpy as np

from ..utils.process import resize, translate_and_crop


class PCBCropOfficialOrigin:
    """
    這是搭配 原paper (official) 的方法使用的。
    """

    def __init__(self, template_size) -> None:
        self.z_size = template_size

    def _make_template(self, img, box, context_amount=0.5, padding=(0, 0, 0)):
        box = box.squeeze()
        # 裁切的公式算法，原始作法要去看 [SiamFC](https://arxiv.org/pdf/1606.09549.pdf)
        # 但其實 SiamCAR 這裡和原論文的作法不太一樣，不過最後結果應該是一樣的...吧？
        # 公式: crop_side = ((w + p) × (h + p)) ^ 1/2
        #                   p = (w + h) / 2
        gt_size = [(box[2] - box[0]), (box[3] - box[1])]
        wc_z = gt_size[1] + context_amount * sum(gt_size)
        hc_z = gt_size[0] + context_amount * sum(gt_size)
        crop_side = np.sqrt(wc_z * hc_z)
        # scale: 縮放比例 (search image 也要做)
        scale = self.z_size / crop_side
        img, box = resize(img, box, scale)

        # x, y 軸的位移距離
        x = self.z_size / 2 - (box[0] + box[2]) / 2
        y = self.z_size / 2 - (box[1] + box[3]) / 2
        img, box, _ = translate_and_crop(
            img, box, translate_px=(x, y), size=self.z_size, padding=padding)
        return img, box, scale

    def get_template(self, img, box, padding=(0, 0, 0)):
        img, box, r = self._make_template(
            img, box, padding=padding)
        return img, box, r

    def get_search(self, img, gt_boxes, z_box, r):
        # 用 template 算出來的 r 來做縮放
        img, gt_boxes = resize(img, gt_boxes, scale=r)
        # template 本身的座標也要修改
        _, z_box = resize(img=None, boxes=z_box, scale=r)
        return img, gt_boxes, z_box

    def get_data(
        self,
        img,
        z_box,
        gt_boxes,
        padding
    ):
        # z_img: (127, 127, 3)
        z_img, _, r = self.get_template(
            img, z_box, padding=padding)
        # x_img: (255, 255, 3)
        x_img, gt_boxes, z_box = self.get_search(
            img, gt_boxes, z_box, r)

        return z_img, x_img, z_box, gt_boxes
