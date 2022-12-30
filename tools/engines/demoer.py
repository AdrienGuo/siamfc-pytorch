import os

import cv2
import ipdb
import torch
from torch.utils.data import DataLoader

from siamfc import SiamFCTemplateMatch
from siamfc.datasets.pcbdataset.pcbdataset import PCBDataset
from utils.file_organizer import create_dir, create_dirs, save_img
from utils.painter import draw_boxes
from siamfc.box_transforms import x1y1x2y2tox1y1wh


class Demoer(object):
    def __init__(self, args) -> None:
        self.args = args
        self.matcher = SiamFCTemplateMatch()

        # Create directory for saving results
        model_name = args.model.split('/')[-1].split('.')[0]
        self.save_dir = os.path.join("./results", "TRI", "tmp", model_name)
        create_dir(self.save_dir)

    def setup_model(self, model_path):
        assert model_path, "ERROR, model_path is empty"
        self.matcher.load_model(model_path)

    def build_dataloader(self, data_path):
        dataset = PCBDataset(self.args, data_path, mode="test")
        assert len(dataset) != 0, "Data is empty"
        print(f"Size of data: {len(dataset)}")
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
        )
        self.dataloader = dataloader

    def start(self) -> None:
        for iter, data in enumerate(self.dataloader):
            img_name = data['img_name'][0]  # 因為 batch=1，只拿第一個就好
            z_img = data['z_img']
            x_img = data['x_img']
            z_box = data['z_box'][0]
            gt_boxes = data['gt_boxes'][0].numpy()

            self.matcher.init(z_img, z_box)
            pred_boxes = self.matcher.match(x_img)

            self._save_result(
                z_img, x_img, z_box, gt_boxes, pred_boxes,
                dir_name=img_name, idx=iter)

            # ipdb.set_trace()

    def _save_result(
        self,
        z_img, x_img, z_box, gt_boxes, pred_boxes,
        dir_name, idx
    ):
        dir = os.path.join(self.save_dir, dir_name)
        create_dir(dir)
        o_dir = os.path.join(dir, "origin")
        z_dir = os.path.join(dir, "template")
        x_dir = os.path.join(dir, "search")
        pred_dir = os.path.join(dir, "pred")
        create_dirs(o_dir, z_dir, x_dir, pred_dir)

        z_img = self._tensor2array(z_img)
        x_img = self._tensor2array(x_img)
        # 注意 boxes 的格式
        gt_boxes = x1y1x2y2tox1y1wh(gt_boxes)
        pred_img = draw_boxes(x_img, gt_boxes, type="gt")
        pred_img = draw_boxes(pred_img, pred_boxes, type="pred")

        self._save_img(z_img, dir=z_dir, idx=idx)
        self._save_img(x_img, dir=x_dir, idx=idx)
        self._save_img(pred_img, dir=pred_dir, idx=idx)

    def _save_img(self, img, dir, idx):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        path = os.path.join(dir, f"{idx}.jpg")
        save_img(img, path)

    def _tensor2array(self, tensor):
        return tensor.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
