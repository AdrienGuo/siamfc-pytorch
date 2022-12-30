import os

import ipdb
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from config.config import Config
from siamfc.datasets.pcbdataset.pcbdataset import PCBDataset
from siamfc.datasets.transforms import PCBTransforms
from siamfc.labels import create_labels
from siamfc.losses import BalancedLoss
from siamfc.models.backbone.backbones import AlexNetV1
from siamfc.models.head.heads import SiamFC
from siamfc.models.model_builder import SeperateNet, SiameseNet
from utils.average_meter import AverageMeter
from utils.file_organizer import create_dir
from utils.wandb import WandB


class Trainer(object):
    def __init__(self, args) -> None:
        # Setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        self.args = args
        # Load config.yaml
        self.cfg = Config(yaml_path="./config/config.yaml")
        self.cfg.update_with_dict(vars(self.args))

        # Setup model
        self.model = SiameseNet(
            backbone=AlexNetV1(),
            head=SiamFC(out_scale=0.001)
        ).to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum
        )
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epochs)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

    def build_dataloaders(self):
        def create_dataloader(dataset):
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=True
            )
            return dataloader

        # setup dataset
        transforms = PCBTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)

        dataset = PCBDataset(
            self.args, self.args.data, mode="train", transforms=transforms)
        test_dataset = PCBDataset(
            self.args, self.args.test_data, mode="test", transforms=transforms)
        assert len(dataset) != 0, "Data is empty"
        print(f"Train data size: {len(dataset)}")
        print(f"Test data size: {len(test_dataset)}")

        self.train_loader = create_dataloader(dataset)
        self.test_loader = create_dataloader(test_dataset)

    def train(self):
        self.model.train()
        model_name = "tmp_mid"

        # Initialize WandB
        wandb = WandB(
            name=model_name, config=self.cfg, init=True)

        # Training
        for epoch in range(self.cfg.epochs):
            print(f"Epoch: [{epoch+1}/{self.cfg.epochs}]")

            train_loss = self.train_one_epoch()
            val_loss = self.validation()
            # Upload loss to WandB
            wandb.upload(
                train_loss.avg, val_loss.avg, epoch=epoch+1
            )

            # Update lr at each epoch
            self.lr_scheduler.step()

            # self._save_checkpoint(epoch+1, dir_name=model_name)

    def train_one_epoch(self):
        self.model.train()

        train_loss = AverageMeter(name="Loss", num=len(self.train_loader))
        for iter, data in enumerate(self.train_loader):
            z_img = data['z_img'].to(self.device)
            x_img = data['x_img'].to(self.device)
            # responses: (B, 1, 15, 15)
            responses = self.model(z_img, x_img)

            # TODO: labels 的作法
            labels = self._create_labels(responses.size())
            # labels = create_labels(
            #     responses.size(), self.cfg).to(self.device)

            # Calculate loss
            loss = self.criterion(responses, labels)
            train_loss.update(val=loss.item(), n=self.cfg.batch_size)
            train_loss.display(type="val", iter=iter+1)

            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Train Summary:")
        train_loss.display(type="avg", iter="finish")
        return train_loss

    def validation(self):
        self.model.eval()

        val_loss = AverageMeter(name="Loss", num=len(self.test_loader))
        with torch.no_grad():
            for iter, data in enumerate(self.test_loader):
                z_img = data['z_img'].to(self.device)
                x_img = data['x_img'].to(self.device)
                # responses: (B, 1, 15, 15)
                responses = self.model(z_img, x_img)

                # TODO: labels 的作法
                labels = self._create_labels(responses.size())
                # labels = create_labels(
                #     responses.size(), self.cfg).to(self.device)

                # calculate loss
                loss = self.criterion(responses, labels)
                val_loss.update(val=loss.item(), n=self.cfg.batch_size)

        print("Validation Summary:")
        val_loss.display(type="avg", iter="finish")
        return val_loss

    def _save_checkpoint(self, epoch, dir_name):
        save_dir = os.path.join("./models", dir_name)
        create_dir(dir=save_dir)
        model_path = os.path.join(save_dir, f"ckpt{epoch}.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Save model to: {model_path}")

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()

        return self.labels
