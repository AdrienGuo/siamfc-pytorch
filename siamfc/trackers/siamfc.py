from __future__ import absolute_import, division, print_function

import os
import sys
import time
from collections import namedtuple

import cv2
import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from got10k.trackers import Tracker
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from utils import create_dir, save_img
from utils.average_meter import AverageMeter
from utils.wandb import WandB

from .. import ops
from ..datasets.datasets import Pair
from ..datasets.utils.transforms import SiamFCTransforms
from ..losses import BalancedLoss
from ..models.backbone.backbones import AlexNetV1
from ..models.head.heads import SiamFC

__all__ = ['TrackerSiamFC']


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class SeperateNet(nn.Module):
    def __init__(self, backbone_z, backbone_x, head):
        super(SeperateNet, self).__init__()
        self.backbone_z = backbone_z
        self.backbone_x = backbone_x
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone_z(z)
        x = self.backbone_x(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        # self.net = SeperateNet(
        #     backbone_z=AlexNetV1(),
        #     backbone_x=AlexNetV1(),
        #     head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)
        
        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 1,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 200,
            'batch_size': 8,
            'num_workers': 10,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0
        }

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        # box: [x1, y1, w, h] -> [cy, cx, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        # 272 = 16 * 17
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        # self.hann_window: (272, 272)
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        # context = 0.5 * (w+h)
        context = self.cfg.context * np.sum(self.target_sz)
        # self.z_sz = ( (w + context) * (h + context) ) ^ (1/2)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)
        
        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        # self.kernel: (1, 256, 6, 6)
        self.kernel = self.net.backbone(z)
        self._save_crop(z, dir="template", name="z.jpg")
    
    @torch.no_grad()
    def update(self, img, frame):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        # x: (scale_num, 3, 255, 255)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        self._save_crop(x, dir="search", name=f"{frame}.jpg")
        
        # responses
        # x: (scale_num, 256, 22, 22)
        x = self.net.backbone(x)
        # responses: (scale_num, 1, 17, 17)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        # responses: (3, 272, 272)
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        # scale_id: 在 scale_num 個裡面，最大的那個數 (選擇維度)
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        # response /= response.sum() + 1e-16
        response /= response.max()
        # response: (272, 272)
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        # disp_in_response: 在 self.upscale_sz，以 (135.5, 135.5) 當作中心點後，位置的比例
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        # disp_in_instance: 改換到 self.scale_sz 上的比例後 (除以 response_up)，再乘上 total_stride 變成實際的位移
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        # box: [x1, y1, w, h]
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
    
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img, f)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :], frame=f)

            ipdb.set_trace()

        return boxes, times
    
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        # z: (B, 3, 127, 127)
        # x: (B, 3, 239, 239)
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            # responses: (B, 1, 15, 15)
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)
        val_dataset = Pair(
            seqs=val_seqs,
            transforms=transforms
        )
        
        # setup train_loader
        train_loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True
        )

        # Initialize WandB
        wandb = WandB(
            name="siamfc", config=self.cfg._asdict(), init=False)
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            epoch_info = {'Train': {}, 'Test': {}}
            train_loss = AverageMeter(name="Loss", num=len(train_loader))
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over train_loader
            for it, batch in enumerate(train_loader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(train_loader), loss))
                train_loss.update(val=loss)
                sys.stdout.flush()
            print("Training Summary:")
            train_loss.display(type="avg", iter="finish")
            
            # Validation
            val_loss = AverageMeter(name="Loss", num=len(val_loader))
            for it, batch in enumerate(val_loader):
                loss = self.train_step(batch, backward=False)
                val_loss.update(val=loss)
            print("Testing Summary:")
            val_loss.display(type="avg", iter="finish")

            # Upload to WandB
            epoch_info['Train']['Loss'] = train_loss.avg
            epoch_info['Test']['Loss'] = val_loss.avg
            wandb.update(info=epoch_info, epoch=epoch+1)
            wandb.upload(commit=True)

            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'ckpt%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
    
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

    def _save_crop(self, img, dir, name):
        img = img.permute(0, 2, 3, 1).cpu().numpy().squeeze()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        filedir = os.path.join("./results", "Crossing", dir)
        create_dir(filedir)
        filename = os.path.join(filedir, name)
        save_img(img, filename)
