import argparse
import os

import ipdb
import numpy as np
import torch
from engines.evaluator import Evaluator
from torch.utils.data import DataLoader

from config.config import Config
from siamfc import SiamFCTemplateMatch
from siamfc.datasets.pcbdataset.pcbdataset import PCBDataset
from siamfc.models.backbone.backbones import AlexNetV1
from siamfc.models.head.heads import SiamFC
from siamfc.models.model_builder import SeperateNet, SiameseNet


def load_model(cfg, model_path):
    # Setup GPU device if available
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    assert model_path, "model_path is empty"
    print(f"Load model from: {model_path}")

    # Setup model
    model = SiameseNet(
        backbone=AlexNetV1(),
        head=SiamFC(cfg.out_scale))
    # model = SeperateNet(
    #     backbone_z=AlexNetV1(),
    #     backbone_x=AlexNetV1(),
    #     head=SiamFC(cfg.out_scale))

    # Load checkpoint
    model.load_state_dict(torch.load(
        model_path, map_location=lambda storage, loc: storage))
    model = model.to(device)
    return model


def build_matcher(model):
    matcher = SiamFCTemplateMatch(model)
    return matcher

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='SiamFC')
    parser.add_argument('--model', type=str, default='', help='path to model')
    parser.add_argument('--part', type=str, default='', help='train / test')
    parser.add_argument('--data', type=str, default='', help='which data')
    parser.add_argument('--data_path', type=str, default='', help='path to data')
    parser.add_argument('--criteria', type=str, default='', help='all / big / mid / small')
    parser.add_argument('--target', type=str, default='', help='one / multi')
    parser.add_argument('--method', type=str, default='', help='official / origin')
    parser.add_argument('--bg', type=str, default='', help='background')
    args = parser.parse_args()

    # Load config.yaml & Combine with args
    cfg = Config(yaml_path="./config/config.yaml")
    cfg.update_with_dict(vars(args))

    # Create evaluator
    evaluator = Evaluator()

    # Load model
    # 先 load model 後，再用 model 去 build matcher
    model = load_model(cfg, cfg.model)
    matcher = build_matcher(model)

    # Build dataloader
    dataset = PCBDataset(
        cfg,
        cfg.data_path,
        method=cfg.method,
        criteria=cfg.criteria,
        bg=cfg.bg,
        mode="test"
    )
    assert len(dataset) != 0, "Data is empty"
    print(f"Data size: {len(dataset)}")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False)

    # Evaluating
    metrics = evaluator.evaluate(
        model=matcher, dataloader=dataloader)
    print(metrics)
