'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/26/2025
'''

import sys, os, argparse, time, logging, wandb
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

import torch, numpy as np
from torch import nn, Tensor, FloatTensor
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import functional as F
from lightning.fabric import Fabric
from torch.utils.data import DataLoader

from systems import InvertedPendulum
from controllers import Controller, ConstantController
from models import MLP, QuadMLP, QuadGoalMLP
from data import TrajDataModule
from engine import InvPend_CLF_Trainer
import utils, utils.loss


def parse_args():
    parser = argparse.ArgumentParser()
    # Data Settings
    parser.add_argument('--n_traj', type=int, default=100)
    parser.add_argument('--T', type=float, default=10.)
    parser.add_argument('--dt', type=float, default=0.01)
    
    # Training Settings
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    # Model Settings
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--layer_num', type=int, default=3)
    parser.add_argument('--lyap_mlp_out', type=int, default=32)
    parser.add_argument('--activation', type=str, default='ELU')
    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--loss_scale', type=float, default=100.0)
    
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    utils.init_seed(args.seed)
    
    dynamic = InvertedPendulum()
    nominal_ctrl = ConstantController(dynamic)
    trainer = InvPend_CLF_Trainer(dynamic, nominal_ctrl, args, ckpt_pth = './checkpoints-test')
    trainer.train()