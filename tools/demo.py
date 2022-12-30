import argparse
import glob
import os

import ipdb
import numpy as np
from engines.demoer import Demoer

from siamfc.trackers.siamfc import TrackerSiamFC

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='SiamFC')
    parser.add_argument('--model', type=str, default='', help='path to model')
    parser.add_argument('--data', type=str, default='', help='path to data')
    parser.add_argument('--criteria', type=str, default='', help='all / big / mid / small')
    parser.add_argument('--target', type=str, default='', help='one / multi')
    parser.add_argument('--method', type=str, default='', help='official / origin')
    parser.add_argument('--bg', type=str, default='', help='background')
    args = parser.parse_args()

    # Start demo
    demoer = Demoer(args)
    demoer.setup_model(model_path=args.model)
    demoer.build_dataloader(data_path=args.data)
    demoer.start()

    # # 官方原本的作法
    # seq_dir = os.path.expanduser('./data/OTB/Crossing/')
    # img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    # anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt', delimiter=' ')
    
    # # net_path = os.path.join('./pretrained', 'official', 'siamfc_alexnet_e50.pth')
    # net_path = os.path.join('./pretrained', 'ckpt150.pth')
    # tracker = TrackerSiamFC(net_path=net_path)
    # tracker.track(img_files, anno[0], visualize=True)
