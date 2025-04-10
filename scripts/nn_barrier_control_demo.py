'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   03/03/2025
'''

import sys, os, argparse, logging
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

from systems import LinearSatellite
from controllers import LQRController
from engine import LinSate_CBF_Trainer
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
    parser.add_argument('--ctrl_type', type=str, default='MLP')
    
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--layer_num', type=int, default=3)
    parser.add_argument('--activation', type=str, default='ELU')
    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--loss_scale', type=float, default=100.0)
    
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    utils.init_seed(args.seed)
    
    dynamic = LinearSatellite()
    nominal_ctrl = LQRController(dynamic)
    trainer = LinSate_CBF_Trainer(dynamic, nominal_ctrl, args, ckpt_pth = './cbf_checkpoints')
    trainer.train()