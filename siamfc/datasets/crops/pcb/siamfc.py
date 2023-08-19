import cv2
import ipdb
import numpy as np

from .... import ops
from ...box_transforms import x1y1x2y2tox1y1wh
from ...utils.process import resize, translate_and_crop
from ...utils.transforms import CenterCrop, Compose, RandomCrop, RandomStretch
from ..crop import crop, crop_with_bg


class PCBCropSiamFC:
    """參考 SiamFC 論文的切法，另外可以用 bg 決定背景多寡。"""

    def __init__(self, template_size, search_size, background, context=0.5) -> None:
        self.z_size = template_size
        self.x_size = search_size
        self.bg = background
        self.context = context

        self.transforms_z = Compose([
            RandomStretch(),
            CenterCrop(search_size - 8),
            RandomCrop(search_size - 2 * 8),
            CenterCrop(template_size),
        ])
        self.transforms_x = Compose([
            RandomStretch(),
            CenterCrop(search_size - 8),
            RandomCrop(search_size - 2 * 8),
        ])

    def z_crop(self, img, box, out_size):
        crop_img = crop(img, box)

        box_size = [(box[2] - box[0]), (box[3] - box[1])]
        wc_z = box_size[1] + self.context * sum(box_size)
        hc_z = box_size[0] + self.context * sum(box_size)
        crop_side = np.sqrt(wc_z * hc_z)
        # scale: 縮放比例 (search image 也要做)
        scale = self.z_size / crop_side
        img, box = resize(crop_img, box, scale)

        # x, y 軸的位移距離
        img_size = img.shape[:2]
        x = out_size / 2 - (img_size[1]) / 2
        y = out_size / 2 - (img_size[0]) / 2
        avg_chans = np.mean(img, axis=(0, 1))
        img, _, _ = translate_and_crop(
            img, box, translate_px=(x, y), size=out_size, padding=avg_chans)
        return img

    def _crop(self, img, box, out_size):
        # convert [x1, y1, w, h] -> [cy, cx, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        center, target_sz = box[:2], box[2:]

        context = self.context * np.sum(target_sz)
        size = np.sqrt(np.prod(target_sz + context))
        size *= out_size / self.z_size

        avg_color = np.mean(img, axis=(0, 1), dtype=float)
        interp = np.random.choice([
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_NEAREST,
            cv2.INTER_LANCZOS4])
        patch = ops.crop_and_resize(
            img, center, size, out_size,
            border_value=avg_color, interp=interp)
        return patch

    def get_data(
        self,
        img,
        z_box,
    ):
        # Crops
        z_box = z_box.squeeze()
        if self.bg == "All":
            z_box = x1y1x2y2tox1y1wh(z_box)
            z_img = self._crop(img, z_box, out_size=self.x_size)
        else:
            # 會使用 bg 參數來決定背景多寡
            z_img = self.z_crop(img, z_box, out_size=self.x_size)
            z_box = x1y1x2y2tox1y1wh(z_box)
        x_img = self._crop(img, z_box, out_size=self.x_size)

        # Transforms
        z_img = self.transforms_z(z_img)
        x_img = self.transforms_x(x_img)
        return z_img, x_img