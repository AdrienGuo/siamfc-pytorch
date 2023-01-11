import os

import ipdb
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from config.config import cfg
from siamfc import SiamFCTemplateMatch
from siamfc.datasets.augmentation.augmentation import PCBAugmentation
from siamfc.datasets.pcbdataset.pcbdataset import PCBDataset
from siamfc.datasets.utils.transforms import PCBTransforms
from siamfc.labels import create_labels
from siamfc.losses import BalancedLoss
from siamfc.models.backbone.backbones import AlexNetV1
from siamfc.models.head.heads import SiamFC
from siamfc.models.model_builder import SeperateNet, SiameseNet
from utils.average_meter import AverageMeter
from utils.file_organizer import create_dir
from utils.wandb import WandB

from .evaluator import Evaluator


class Trainer(object):
    def __init__(self, args) -> None:
        # Setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        self.args = args
        # Load config.yaml & Combine with args
        cfg.merge_from_file("./config/config.yaml")
        cfg.update(vars(args))
        self.cfg = cfg

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

        # Setup evaluator
        self.evaluator = Evaluator()

    def build_dataloaders(self):
        def create_dataloader(dataset, batch_size, shuffle, num_workers):
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True
            )
            return dataloader

        # setup transforms
        transforms = PCBTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)

        # datasets arguments
        train_args = {
            'data_path': self.cfg.data,
            'method': self.cfg.method,
            'criteria': self.cfg.criteria,
            'bg': self.cfg.bg,
            'target': self.cfg.target,
        }

        data_augmentations = {
            'train': {
                'template': PCBAugmentation(self.cfg.train.template),
                'search': PCBAugmentation(self.cfg.train.search),
            },
            'test': {
                
            }
        }
        dataset = PCBDataset(train_args, mode="train", augmentations=data_augmentations['train'], transforms=transforms)
        test_dataset = dataset
        eval_dataset = dataset
        test_eval_dataset = dataset

        # test_dataset = PCBDataset(
        #     self.cfg, self.cfg.test_data,
        #     method=self.cfg.method,
        #     criteria=self.cfg.criteria,
        #     bg=self.cfg.bg,
        #     mode="test",
        #     transforms=transforms)
        # eval_dataset = PCBDataset(
        #     self.cfg, self.cfg.data,
        #     method=self.cfg.eval_method,
        #     criteria=self.cfg.eval_criteria,
        #     bg=self.cfg.eval_bg,
        #     mode="test")
        # test_eval_dataset = PCBDataset(
        #     self.cfg, self.cfg.test_data,
        #     method=self.cfg.eval_method,
        #     criteria=self.cfg.eval_criteria,
        #     bg=self.cfg.eval_bg,
        #     mode="test")
        assert len(dataset) != 0, "Data is empty"
        print(f"Train data size: {len(dataset)}")
        print(f"Test data size: {len(test_dataset)}")
        print(f"Train Eval data size: {len(eval_dataset)}")
        print(f"Test Eval data size: {len(test_eval_dataset)}")
        self.cfg.update({'Train Data Size': len(dataset)})
        self.cfg.update({'Test Data Size': len(test_dataset)})
        self.cfg.update({'Train Eval Data Size': len(eval_dataset)})
        self.cfg.update({'Test Eval Data Size': len(test_eval_dataset)})

        # dataloaders
        self.train_loader = create_dataloader(dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)
        self.test_loader = create_dataloader(test_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)
        self.eval_loader = create_dataloader(eval_dataset, batch_size=1, shuffle=False, num_workers=8)
        self.eval_test_loader = create_dataloader(test_eval_dataset, batch_size=1, shuffle=False, num_workers=8)

    def train(self):
        self.model.train()
        model_name = "Flip_all_all"

        # Initialize WandB
        wandb = WandB(
            name=model_name, config=self.cfg, init=False)

        # Training
        for epoch in range(self.cfg.epochs):
            print(f"Epoch: [{epoch+1}/{self.cfg.epochs}]")
            # Record epoch info
            epoch_info = {'Train': {}, 'Test': {}}

            train_loss = self.train_one_epoch()
            test_loss = self.validation()
            epoch_info['Train'].update(train_loss)
            epoch_info['Test'].update(test_loss)

            # Evaluate
            if (epoch == 0) or (epoch + 1 == self.cfg.epochs) \
                    or ((epoch + 1) % self.cfg.eval_freq == 0):
                train_metrics = self._evaluate(self.model, self.eval_loader)
                test_metrics = self._evaluate(self.model, self.eval_test_loader)
                epoch_info['Train'].update(train_metrics)
                epoch_info['Test'].update(test_metrics)

            # Upload epoch_info to WandB
            wandb.update(info=epoch_info, epoch=epoch+1)
            wandb.upload(commit=True)

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
        return {
            'Loss': train_loss.avg
        }

    def validation(self):
        self.model.eval()

        test_loss = AverageMeter(name="Loss", num=len(self.test_loader))
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
                test_loss.update(val=loss.item(), n=self.cfg.batch_size)

        print("Validation Summary:")
        test_loss.display(type="avg", iter="finish")
        return {
            'Loss': test_loss.avg
        }

    def _evaluate(self, model, dataloader):
        # Load matcher model
        matcher = SiamFCTemplateMatch(model=model)
        metrics = self.evaluator.evaluate(matcher, dataloader)
        return metrics

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
