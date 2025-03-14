import os, argparse, logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

import torch, numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import functional as F
from lightning.fabric import Fabric

from systems import CtrlAffSys
from controllers import Controller
from certificates import NNLyapunov, NNBarrier
from models import MLP
from data import TrajDataModule
import utils
from utils.loss import LyapNNLoss, BarrierNNLoss


class Certif_Ctrl_Trainer:
    def __init__(
        self,
        dynamic: CtrlAffSys,
        nominal_controller: Controller,
        args: argparse.Namespace,
        ckpt_pth: str = './checkpoints',
    ):
        utils.info_args_table(args)
        self.dynamic, self.nominal_controller = dynamic, nominal_controller
        self.args = args
        self.make_ckpt_dir(ckpt_pth)
        
        self.nn_ctrl, self.certif = self.create_models()
        self.loss_fn = self.create_loss_fn()
        
        self.ctrl_optim = Adam(self.nn_ctrl.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
        self.certif_optim = Adam(self.certif.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
        
        self.ctrl_scheduler = CosineAnnealingLR(self.ctrl_optim, self.args.n_epochs, eta_min = 0.1 * self.args.lr)
        self.certif_scheduler = CosineAnnealingLR(self.certif_optim, self.args.n_epochs, eta_min = 0.1 * self.args.lr)
        
        self.fabric = Fabric(accelerator = 'cuda', devices = [self.args.cuda,],)
        self.nn_ctrl, self.ctrl_optim = self.fabric.setup(self.nn_ctrl, self.ctrl_optim)
        self.certif, self.certif_optim = self.fabric.setup(self.certif, self.certif_optim)
        
        self.train_loader, self.val_loader = self.get_dataloaders()
        
        
    def train(self):
        best_epoch, min_val_loss = 0, float('inf')
        for epoch in range(self.args.n_epochs):
            train_res = self.train_step()
            val_res = self.validate_step()
            
            if val_res['total_loss'] < min_val_loss:
                min_val_loss, best_epoch = val_res['total_loss'], epoch
                utils.save_checkpoint(self.nn_ctrl, self.ctrl_optim, epoch, f'{self.ctrl_ckpt_pth}/best_ctrl.pt')
                utils.save_checkpoint(self.certif, self.certif_optim, epoch, f'{self.certif_ckpt_pth}/best_{self.args.certif_type}.pt')
                
            logging.info(f'Epoch {epoch:4d} - Train: {train_res["total_loss"]:.4e} | Ctrl MSE: {train_res["ctrl_mse"]:.4e} | Goal Loss: {train_res["goal_loss"]:.4e} | Deriv Num: {train_res["derive"]["num_loss"]:.4e} | Deriv Ana: {train_res["derive"]["ana_loss"]:.4e} | Deriv Diff: {train_res["derive"]["diff_loss"]:.4e} | Best Epoch: {best_epoch:4d}')
            logging.info(f'Epoch {epoch:4d} - Val  : {val_res["total_loss"]:.4e} | Ctrl MSE: {val_res["ctrl_mse"]:.4e} | Goal Loss: {val_res["goal_loss"]:.4e} | Deriv Num: {val_res["derive"]["num_loss"]:.4e} | Deriv Ana: {val_res["derive"]["ana_loss"]:.4e} | Deriv Diff: {val_res["derive"]["diff_loss"]:.4e} | Best Epoch: {best_epoch:4d}')
            logging.info('-' * 150)
            
            if (epoch + 1) % 500 == 0:
                torch.save(self.nn_ctrl.state_dict(), f'{self.ctrl_ckpt_pth}/epoch{epoch}.pt')
                torch.save(self.certif.state_dict(), f'{self.certif_ckpt_pth}/epoch{epoch}.pt')
    
    
    def train_step(self):
        total_loss_lst, ctrl_mse_lst, goal_loss_lst = [], [], []
        deriv_diff_lst, deriv_num_lst, deriv_ana_lst = [], [], []
        
        for curr_xs, curr_us, next_xs in self.train_loader:
            # Control loss - Restrict the control to the nominal controller
            pred_us = self.nn_ctrl(curr_xs)
            ctrl_mse = F.mse_loss(pred_us, curr_us)
            ctrl_mse_lst.append(ctrl_mse.item())
            
            # Certificate loss
            pred_next_xs = next_xs + self.args.dt * pred_us @ self.dynamic.B.T.to(pred_us.device)
            goal_loss, deriv_num_loss, deriv_ana_loss, diff_loss = self.loss_fn(curr_xs, pred_next_xs)
            goal_loss_lst.append(goal_loss.item())
            deriv_num_lst.append(deriv_num_loss.item())
            deriv_ana_lst.append(deriv_ana_loss.item())
            deriv_diff_lst.append(diff_loss.item())
            
            total_loss = ctrl_mse + self.args.loss_scale * (goal_loss + deriv_num_loss + deriv_ana_loss + diff_loss)
            total_loss_lst.append(total_loss.item())
            
            self.certif_optim.zero_grad()
            self.ctrl_optim.zero_grad()
            self.fabric.backward(total_loss)
            self.certif_optim.step()
            self.ctrl_optim.step()
            self.ctrl_scheduler.step()
            self.certif_scheduler.step()
        
        return {
            'total_loss': np.mean(total_loss_lst),
            'ctrl_mse': np.mean(ctrl_mse_lst),
            'goal_loss': np.mean(goal_loss_lst),
            'derive': {
                'num_loss': np.mean(deriv_num_lst),
                'ana_loss': np.mean(deriv_ana_lst),
                'diff_loss': np.mean(deriv_diff_lst),
            }
        }
    
    
    @torch.no_grad()
    def validate_step(self):
        total_loss_lst, ctrl_mse_lst, goal_loss_lst = [], [], []
        deriv_diff_lst, deriv_num_lst, deriv_ana_lst = [], [], []
        
        for curr_xs, curr_us, next_xs in self.val_loader:
            # Control loss - Restrict the control to the nominal controller
            pred_us = self.nn_ctrl(curr_xs)
            ctrl_mse = F.mse_loss(pred_us, curr_us)
            ctrl_mse_lst.append(ctrl_mse.item())
            
            # Certificate loss
            pred_next_xs = next_xs + self.args.dt * pred_us @ self.dynamic.B.T.to(pred_us.device)
            goal_loss, deriv_num_loss, deriv_ana_loss, diff_loss = self.loss_fn(curr_xs, pred_next_xs)
            goal_loss_lst.append(goal_loss.item())
            deriv_num_lst.append(deriv_num_loss.item())
            deriv_ana_lst.append(deriv_ana_loss.item())
            deriv_diff_lst.append(diff_loss.item())
            
            total_loss = ctrl_mse + self.args.loss_scale * (goal_loss + deriv_num_loss + deriv_ana_loss + diff_loss)
            total_loss_lst.append(total_loss.item())
        
        return {
            'total_loss': np.mean(total_loss_lst),
            'ctrl_mse': np.mean(ctrl_mse_lst),
            'goal_loss': np.mean(goal_loss_lst),
            'derive': {
                'num_loss': np.mean(deriv_num_lst),
                'ana_loss': np.mean(deriv_ana_lst),
                'diff_loss': np.mean(deriv_diff_lst),
            }
        }
    
    
    def create_models(self):
        n_dim, n_ctrl = self.dynamic.n_dim, self.dynamic.n_control
        ctrl = MLP(n_dim, n_ctrl, **vars(self.args))
        if self.args.certif_type == 'lyapunov':
            certif = NNLyapunov(
                self.dynamic, self.nominal_controller, self.args.lamb,
                nn_type = 'QuadGoalMLP', nn_kwargs = {
                    'mlp_output_size': self.args.hidden_size,
                    'hidden_size': self.args.hidden_size,
                    'layer_num': self.args.layer_num,
                    'activation': self.args.activation,
                }
            )
        elif self.args.certif_type == 'barrier':
            certif = NNBarrier(
                self.dynamic, self.nominal_controller, self.args.lamb,
                nn_type = 'MLP', nn_kwargs = vars(self.args)
            )
        else:
            raise ValueError(f'Invalid certificate type: {self.args.certif_type}')
        
        return ctrl, certif
    
    
    def create_loss_fn(self):
        if self.args.certif_type == 'lyapunov':
            loss_fn = LyapNNLoss(self.certif, self.args.lamb, self.args.dt, torch.device(f'cuda:{self.args.cuda}'))
        elif self.args.certif_type == 'barrier':
            loss_fn = BarrierNNLoss(self.certif, self.args.lamb, self.args.dt, torch.device(f'cuda:{self.args.cuda}'))
        else:
            raise ValueError(f'Invalid certificate type: {self.args.certif_type}')
        return loss_fn
    
    
    def make_ckpt_dir(self, ckpt_pth: str):
        self.ctrl_ckpt_pth = f'{ckpt_pth}/ctrl'
        self.certif_ckpt_pth = f'{ckpt_pth}/{self.args.certif_type}'
        os.makedirs(self.ctrl_ckpt_pth, exist_ok = True)
        os.makedirs(self.certif_ckpt_pth, exist_ok = True)
    
    
    def get_dataloaders(self):
        dm = TrajDataModule(self.dynamic, self.nominal_controller, **vars(self.args))
        dm.setup()
        return self.fabric.setup_dataloaders(
            dm.train_dataloader(), dm.val_dataloader()
        )