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
from controllers import ConstantController
from models import MLP, QuadNN
from data import TrajDataModule
import utils, utils.loss


class NNDynamic(nn.Module):
    def __init__(self, dynamic_nn: MLP, ctrl_affine_mat: Tensor):
        super().__init__()
        self.dynamic_nn = dynamic_nn
        self.ctrl_affine_mat = nn.Parameter(
            ctrl_affine_mat, # n_dim x n_ctrl
            requires_grad = False
        )
        
    def forward(self, xs: Tensor, us: Tensor) -> Tensor:
        return self.dynamic_nn(xs) + us @ self.ctrl_affine_mat.T
    
    
class NNCertifCtrl(nn.Module):
    def __init__(self, policy_nn: MLP, lyap_nn: QuadNN, lamb: float):
        super().__init__()
        self.policy_nn = policy_nn
        self.lyap_nn = lyap_nn
        self.lamb = lamb
        
    def forward(self, xs: Tensor) -> Tensor:
        return self.policy_nn(xs)
    
    def lyap_goal_loss(self, goal_point: Tensor) -> Tensor:
        return self.lyap_nn(goal_point).mean()
    
    def lyap_deriv_loss(self, curr_xs: Tensor, next_xs: Tensor) -> Tensor:
        curr_vs = self.lyap_nn(curr_xs)
        next_vs = self.lyap_nn(next_xs)
        return F.relu(curr_vs - self.lamb * next_vs).mean()
        
        
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
    parser.add_argument('--lamb', type=float, default=0.1)
    
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def get_dataloaders(dynamic, controller, args):
    dm = TrajDataModule(dynamic, controller, **vars(args))
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    return train_loader, val_loader


def train_dynamic(
    nn_dynamic: NNDynamic,
    dynamic_optim: Optimizer,
    dynamic_scheduler: CosineAnnealingLR,
    train_loader: DataLoader,
    fabric: Fabric,
):
    dynamic_loss_lst = []
    for curr_xs, curr_us, next_xs in train_loader:
        pred_next_xs = nn_dynamic(curr_xs, curr_us)
        loss = F.mse_loss(pred_next_xs, next_xs)
        fabric.backward(loss)
        dynamic_optim.step()
        dynamic_scheduler.step()
        dynamic_loss_lst.append(loss.item())
    return np.mean(dynamic_loss_lst)


def train_ctrl(
    nn_dynamic: NNDynamic,
    nn_ctrl: NNCertifCtrl,
    goal_point: Tensor,
    ctrl_optim: Optimizer,
    ctrl_scheduler: CosineAnnealingLR,
    train_loader: DataLoader,
    fabric: Fabric,
):
    ctrl_loss_lst, ctrl_mse_lst, goal_loss_lst, deriv_loss_lst = [], [], [], []
    for curr_xs, curr_us, _ in train_loader:
        pred_us = nn_ctrl(curr_xs)
        ctrl_mse_loss = F.mse_loss(pred_us, curr_us)
        ctrl_mse_lst.append(ctrl_mse_loss.item())
        
        goal_loss = nn_ctrl.lyap_goal_loss(goal_point.to(curr_xs.device))
        goal_loss_lst.append(goal_loss.item())
        
        pred_next_xs = nn_dynamic(curr_xs, pred_us)
        deriv_loss = nn_ctrl.lyap_deriv_loss(curr_xs, pred_next_xs)
        deriv_loss_lst.append(deriv_loss.item())
        
        ctrl_loss = 100 * ctrl_mse_loss + 100 * goal_loss + 100 * deriv_loss
        ctrl_loss_lst.append(ctrl_loss.item())
        
        ctrl_optim.zero_grad()
        fabric.backward(ctrl_loss)
        ctrl_optim.step()
        ctrl_scheduler.step()
        
    ctrl_loss = np.mean(ctrl_loss_lst)
    ctrl_mse = np.mean(ctrl_mse_lst)
    goal_loss = np.mean(goal_loss_lst)
    deriv_loss = np.mean(deriv_loss_lst)
    return ctrl_loss, ctrl_mse, goal_loss, deriv_loss


@torch.no_grad()
def validate_dynamic(
    nn_dynamic: NNDynamic,
    val_loader: DataLoader,
):
    dynamic_loss_lst = []
    for curr_xs, curr_us, next_xs in val_loader:
        pred_next_xs = nn_dynamic(curr_xs, curr_us)
        loss = F.mse_loss(pred_next_xs, next_xs)
        dynamic_loss_lst.append(loss.item())
    return np.mean(dynamic_loss_lst)


@torch.no_grad()
def validate_ctrl(
    nn_dynamic: NNDynamic,
    nn_ctrl: NNCertifCtrl,
    goal_point: Tensor,
    val_loader: DataLoader,
):
    ctrl_loss_lst, ctrl_mse_lst, goal_loss_lst, deriv_loss_lst = [], [], [], []
    for curr_xs, curr_us, _ in val_loader:
        pred_us = nn_ctrl(curr_xs)
        ctrl_mse_loss = F.mse_loss(pred_us, curr_us)
        ctrl_mse_lst.append(ctrl_mse_loss.item())
        
        goal_loss = nn_ctrl.lyap_goal_loss(goal_point.to(curr_xs.device))
        goal_loss_lst.append(goal_loss.item())
        
        pred_next_xs = nn_dynamic(curr_xs, pred_us)
        deriv_loss = nn_ctrl.lyap_deriv_loss(curr_xs, pred_next_xs)
        deriv_loss_lst.append(deriv_loss.item())
        
        ctrl_loss = 100 * ctrl_mse_loss + 100 * goal_loss + 100 * deriv_loss
        ctrl_loss_lst.append(ctrl_loss.item())
        
    ctrl_loss = np.mean(ctrl_loss_lst)
    ctrl_mse = np.mean(ctrl_mse_lst)
    goal_loss = np.mean(goal_loss_lst)
    deriv_loss = np.mean(deriv_loss_lst)
    return ctrl_loss, ctrl_mse, goal_loss, deriv_loss


if __name__ == '__main__':
    args = parse_args()
    ckpt_pth = f'{os.path.dirname(SCRIPT_DIR)}/checkpoints'
    os.makedirs(ckpt_pth, exist_ok = True)
    utils.info_args_table(args)
    utils.init_seed(args.seed)
    
    wandb.init(project = 'inverted_pendulum_nn_demo', config = vars(args))
    
    # Initialize system, controller and fabric
    dynamic = InvertedPendulum()
    controller = ConstantController(dynamic)
    n_dim, n_ctrl = dynamic.n_dim, dynamic.n_control
    fabric = Fabric(accelerator = 'cuda', devices = [args.cuda,])
    fabric.launch()
    
    # Initialize DataLoaders
    train_loader, val_loader = get_dataloaders(dynamic, controller, args)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    
    # Initialize Models
    dynamic_nn = MLP(n_dim, n_dim, **vars(args))
    policy_nn = MLP(n_dim, n_ctrl, **vars(args))
    lyap_nn = QuadNN(n_dim, args.lyap_mlp_out, **vars(args))
    
    nn_dynamic = NNDynamic(dynamic_nn, FloatTensor(([[0.0], [1.0]])))
    nn_ctrl = NNCertifCtrl(policy_nn, lyap_nn, args.lamb)
    
    # Initialize Optimizers
    dynamic_optim: Optimizer = Adam(nn_dynamic.parameters(), lr = args.lr, weight_decay = args.weight_decay) # 
    dynamic_scheduler = CosineAnnealingLR(dynamic_optim, T_max = args.n_epochs, eta_min = 1e-6)
    ctrl_optim: Optimizer = Adam(nn_ctrl.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    ctrl_scheduler = CosineAnnealingLR(ctrl_optim, T_max = args.n_epochs, eta_min = 1e-6)
    
    nn_dynamic, dynamic_optim = fabric.setup(nn_dynamic, dynamic_optim)
    nn_ctrl, ctrl_optim = fabric.setup(nn_ctrl, ctrl_optim)
    
    # training loop
    for epoch in range(args.n_epochs):
        # train models
        start_time = time.time()
        train_dynamic_loss = train_dynamic(nn_dynamic, dynamic_optim, dynamic_scheduler, train_loader, fabric)
        train_ctrl_loss, train_ctrl_mse, train_goal_loss, train_deriv_loss = train_ctrl(
            nn_dynamic, nn_ctrl, dynamic.goal_point, ctrl_optim, ctrl_scheduler, train_loader, fabric
        )
        logging.info(f'Train | Epoch {epoch:3} | Dynamic Loss: {train_dynamic_loss:.3e} | Ctrl Loss: {train_ctrl_loss:.3e} | Ctrl MSE: {train_ctrl_mse:.3e} | Goal Loss: {train_goal_loss:.3e} | Deriv Loss: {train_deriv_loss:.3e}')
        
        # validate models
        val_dynamic_loss = validate_dynamic(nn_dynamic, val_loader)
        val_ctrl_loss, val_ctrl_mse, val_goal_loss, val_deriv_loss = validate_ctrl(
            nn_dynamic, nn_ctrl, dynamic.goal_point, val_loader
        )
        logging.info(f'Val   | Epoch {epoch:3} | Dynamic Loss: {val_dynamic_loss:.3e} | Ctrl Loss: {val_ctrl_loss:.3e} | Ctrl MSE: {val_ctrl_mse:.3e} | Goal Loss: {val_goal_loss:.3e} | Deriv Loss: {val_deriv_loss:.3e}')
        end_time = time.time()
        
        logging.info(f'Time taken: {end_time - start_time:.2f} seconds | Dynamic LR: {dynamic_scheduler.get_last_lr()[0]:.6e} | Ctrl LR: {ctrl_scheduler.get_last_lr()[0]:.6e}')
        logging.info('-' * 150)
        
        wandb.log({
            'train': {
                'dynamic_loss': train_dynamic_loss,
                'ctrl_loss': train_ctrl_loss,
                'ctrl_mse': train_ctrl_mse,
                'goal_loss': train_goal_loss,
                'deriv_loss': train_deriv_loss,
            },
            'val': {    
                'dynamic_loss': val_dynamic_loss,   
                'ctrl_loss': val_ctrl_loss,
                'ctrl_mse': val_ctrl_mse,
                'goal_loss': val_goal_loss,
                'deriv_loss': val_deriv_loss,
            },
            'time': end_time - start_time,
            'epoch': epoch,
            'lr': {
                'dynamic': dynamic_scheduler.get_last_lr()[0],
                'ctrl': ctrl_scheduler.get_last_lr()[0],
            },
        })
        
        # save models
        utils.save_checkpoint(nn_dynamic, dynamic_optim, epoch, f'{ckpt_pth}/inverted_pendulum_nn_demo-epoch{epoch}.pt')
        utils.save_checkpoint(nn_ctrl, ctrl_optim, epoch, f'{ckpt_pth}/inverted_pendulum_nn_demo-epoch{epoch}.pt')
        
            