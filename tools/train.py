import argparse
import os

import ipdb
from got10k.datasets import *

from siamfc import TrackerSiamFC
from tools.engines.trainer import Trainer

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='SiamFC training arguments')
    parser.add_argument('--data', type=str, default='', help='path to train data')
    parser.add_argument('--criteria', type=str, default='', help='all / big / mid / small')
    parser.add_argument('--method', type=str, default='', help='official / origin')
    parser.add_argument('--bg', type=str, default='', help='background')
    parser.add_argument('--test_data', type=str, default='', help='path to test data')
    parser.add_argument('--eval_criteria', type=str, default='', help='all / big / mid / small')
    parser.add_argument('--eval_method', type=str, default='', help='official_origin')
    parser.add_argument('--eval_bg', type=str, default='', help='background')
    parser.add_argument('--target', type=str, default='', help='one / multi')
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.build_dataloaders()
    trainer.train()

    # 官方做法
    # root_dir = os.path.expanduser('./data/GOT-10k')
    # seqs = GOT10k(root_dir, subset='train', return_meta=True)
    # val_seqs = GOT10k(root_dir, subset='val', return_meta=True)

    # tracker = TrackerSiamFC()
    # tracker.train_over(seqs, val_seqs=val_seqs)
