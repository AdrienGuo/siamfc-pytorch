import os

import cv2
import ipdb
import torch
from torch.utils.data import DataLoader

from config.config import Config
from siamfc import SiamFCTemplateMatch
from siamfc.box_transforms import x1y1x2y2tox1y1wh
from siamfc.datasets.pcbdataset import get_pcbdataset
from siamfc.models.backbone.backbones import AlexNetV1
from siamfc.models.head.heads import SiamFC
from siamfc.models.model_builder import SeperateNet, SiameseNet
from utils.file_organizer import create_dir, create_dirs, save_img
from utils.painter import draw_boxes


class Demoer(object):
    def __init__(self, args) -> None:
        # Setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        self.args = args
        # Load config.yaml & Combine with args
        self.cfg = Config(yaml_path="./config/config.yaml")
        self.cfg.update_with_dict(vars(self.args))

        # 先 load model 後，再用 model 去 build matcher
        model = self.load_model(args.model)
        self.matcher = self.build_matcher(model)

        # Create directory for saving results
        model_name = args.model.split('/')[-2]
        model_ckpt = args.model.split('/')[-1].split('.')[0]
        self.save_dir = os.path.join(
            "./results", "TRI",
            args.part, args.data, args.method,
            model_name, model_ckpt)
        create_dir(self.save_dir)

    def load_model(self, model_path):
        assert model_path, "model_path is empty"
        print(f"Load model from: {model_path}")

        # Setup model
        model = SiameseNet(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        # model = SeperateNet(
        #     backbone_z=AlexNetV1(),
        #     backbone_x=AlexNetV1(),
        #     head=SiamFC(self.cfg.out_scale))

        # Load checkpoint
        model.load_state_dict(torch.load(
            model_path, map_location=lambda storage, loc: storage))
        model = model.to(self.device)

        return model

    def build_matcher(self, model):
        matcher = SiamFCTemplateMatch(model)
        return matcher

    def build_dataloader(self, data_path):
        pcb_dataset = get_pcbdataset(self.args.method)
        dataset = pcb_dataset(
            self.cfg, data_path, self.args.method,
            criteria=self.args.criteria,
            bg=self.cfg.bg,
            mode="test",
        )
        assert len(dataset) != 0, "Data is empty"
        print(f"Data size: {len(dataset)}")
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
            z_box = data['z_box'][0].numpy()
            gt_boxes = data['gt_boxes'][0].numpy()

            self.matcher.init(z_img, z_box)
            pred_boxes = self.matcher.match(x_img)

            self._save_result(
                z_img, x_img, z_box, gt_boxes, pred_boxes,
                dir_name=img_name, idx=iter)

            # ipdb.set_trace()

        # TODO: Evaluating

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
        pred_img = draw_boxes(x_img, gt_boxes, type="gt")
        pred_img = draw_boxes(pred_img, pred_boxes, type="pred")

        self._save_img(z_img, dir=z_dir, idx=idx)
        self._save_img(x_img, dir=x_dir, idx=idx)
        self._save_img(pred_img, dir=pred_dir, idx=idx)

    def _save_img(self, img, dir, idx):
        path = os.path.join(dir, f"{idx}.jpg")
        save_img(img, path)

    def _tensor2array(self, tensor):
        return tensor.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
