import sys, os, argparse, time, logging, wandb
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

import torch, numpy as np
from torch import nn, Tensor, FloatTensor
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import functional as F
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
from torch.utils.data import DataLoader

from systems import InvertedPendulum
from controllers import Controller, ConstantController
from certificates import NNLyapunov
from models import MLP, QuadMLP, QuadGoalMLP
from data import TrajDataModule
import utils
from utils.loss import LyapNNLoss


class InvPend_CLF_Trainer:
    def __init__(
        self,
        dynamic: InvertedPendulum,
        nominal_controller: Controller,
        args: argparse.Namespace,
        ckpt_pth: str = './checkpoints',
    ):
        self.dynamic, self.nominal_controller = dynamic, nominal_controller
        self.args = args
        
        self.ckpt_pth = ckpt_pth
        self.lyap_ckpt_pth = f'{ckpt_pth}/inv_pend_lyap-lamb{args.lamb}'
        self.ctrl_ckpt_pth = f'{ckpt_pth}/inv_pend_ctrl-lamb{args.lamb}'
        os.makedirs(self.lyap_ckpt_pth, exist_ok = True)
        os.makedirs(self.ctrl_ckpt_pth, exist_ok = True)
        
        self.fabric = Fabric(accelerator = 'cuda', devices = [args.cuda,],)
        
        self.nn_ctrl, self.lyapunov = self.create_models()
        self.ctrl_mat = FloatTensor(([[0.0], [1.0]]))
        
        self.ctrl_optim = Adam(self.nn_ctrl.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        self.ctrl_scheduler = CosineAnnealingLR(self.ctrl_optim, args.n_epochs, eta_min = 0.1 * args.lr)
        
        self.lyap_optim = Adam(self.lyapunov.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        self.lyap_scheduler = CosineAnnealingLR(self.lyap_optim, args.n_epochs, eta_min = 0.1 * args.lr)
        
        self.nn_ctrl, self.ctrl_optim = self.fabric.setup(self.nn_ctrl, self.ctrl_optim)  
        self.lyapunov, self.lyap_optim = self.fabric.setup(self.lyapunov, self.lyap_optim)
        
        self.train_loader, self.val_loader = self.get_dataloaders()
        self.loss_fn = LyapNNLoss(self.lyapunov, self.args.lamb, self.args.dt, torch.device(f'cuda:{self.args.cuda}'))
        
        self.nn_ctrl: MLP
        self.lyapunov: NNLyapunov
    
    def train(self):
        best_epoch, min_val_loss = 0, float('inf')
        for epoch in range(self.args.n_epochs):
            train_ctrl_mse, train_goal_loss, train_deriv_num_loss, train_deriv_ana_loss, train_deriv_diff_loss, train_total_loss = self.train_step()
            val_ctrl_mse, val_goal_loss, val_deriv_num_loss, val_deriv_ana_loss, val_deriv_diff_loss, val_total_loss = self.validate_step()
            
            if val_total_loss < min_val_loss:
                min_val_loss, best_epoch = val_total_loss, epoch
                utils.save_checkpoint(self.nn_ctrl, self.ctrl_optim, epoch, f'{self.ctrl_ckpt_pth}/best_ctrl.pt')
                utils.save_checkpoint(self.lyapunov, self.lyap_optim, epoch, f'{self.lyap_ckpt_pth}/best_lyap.pt')
            
            logging.info(f'Epoch {epoch:4d} - Train: {train_total_loss:.4e} | MSE: {train_ctrl_mse:.4e} | Goal: {train_goal_loss:.4e} | Deriv Num: {train_deriv_num_loss:.4e} | Deriv Ana: {train_deriv_ana_loss:.4e} | Deriv Diff: {train_deriv_diff_loss:.4e} | Best Epoch: {best_epoch:4d}')
            logging.info(f'Epoch {epoch:4d} - Val  : {val_total_loss:.4e} | MSE: {val_ctrl_mse:.4e} | Goal: {val_goal_loss:.4e} | Deriv Num: {val_deriv_num_loss:.4e} | Deriv Ana: {val_deriv_ana_loss:.4e} | Deriv Diff: {val_deriv_diff_loss:.4e} | Best Epoch: {best_epoch:4d}')
            logging.info('-' * 150)
            
            if epoch % 100 == 0:
                torch.save(self.nn_ctrl.state_dict(), f'{self.ctrl_ckpt_pth}/epoch{epoch}.pt')
                torch.save(self.lyapunov.state_dict(), f'{self.lyap_ckpt_pth}/epoch{epoch}.pt')
        
    def train_step(self):
        total_loss_lst, ctrl_mse_lst, goal_loss_lst = [], [], []
        deriv_diff_lst, deriv_num_lst, deriv_ana_lst = [], [], []
        for curr_xs, curr_us, next_xs in self.train_loader:
            # Control loss - Restrict the control to the nominal controller
            pred_us = self.nn_ctrl(curr_xs)
            ctrl_mse = F.mse_loss(pred_us, curr_us)
            ctrl_mse_lst.append(ctrl_mse.item())
            
            pred_next_xs = next_xs + self.args.dt * pred_us @ self.ctrl_mat.T.to(pred_us.device)
            goal_loss, deriv_num_loss, deriv_ana_loss, diff_loss = self.loss_fn(curr_xs, pred_next_xs)
            goal_loss_lst.append(goal_loss.item())
            deriv_num_lst.append(deriv_num_loss.item())
            deriv_ana_lst.append(deriv_ana_loss.item())
            deriv_diff_lst.append(diff_loss.item())
            
            total_loss = ctrl_mse + self.args.loss_scale * (goal_loss + deriv_num_loss + deriv_ana_loss + diff_loss)
            total_loss_lst.append(total_loss.item())
            
            self.lyap_optim.zero_grad()
            self.ctrl_optim.zero_grad()
            self.fabric.backward(total_loss)
            self.lyap_optim.step()
            self.ctrl_optim.step()
            self.ctrl_scheduler.step()
            self.lyap_scheduler.step()
        
        ctrl_mse = np.mean(ctrl_mse_lst)
        goal_loss = np.mean(goal_loss_lst)
        deriv_num_loss = np.mean(deriv_num_lst)
        deriv_ana_loss = np.mean(deriv_ana_lst)
        deriv_diff_loss = np.mean(deriv_diff_lst)
        total_loss = np.mean(total_loss_lst)
        
        return ctrl_mse, goal_loss, deriv_num_loss, deriv_ana_loss, deriv_diff_loss, total_loss
    
    @torch.no_grad()
    def validate_step(self):
        total_loss_lst, ctrl_mse_lst, goal_loss_lst = [], [], []
        deriv_diff_lst, deriv_num_lst, deriv_ana_lst = [], [], []
        for curr_xs, curr_us, next_xs in self.train_loader:
            # Control loss - Restrict the control to the nominal controller
            pred_us = self.nn_ctrl(curr_xs)
            ctrl_mse = F.mse_loss(pred_us, curr_us)
            ctrl_mse_lst.append(ctrl_mse.item())
            
            pred_next_xs = next_xs + self.args.dt * pred_us @ self.ctrl_mat.T.to(pred_us.device)
            goal_loss, deriv_num_loss, deriv_ana_loss, diff_loss = self.loss_fn(curr_xs, pred_next_xs)
            goal_loss_lst.append(goal_loss.item())
            deriv_num_lst.append(deriv_num_loss.item())
            deriv_ana_lst.append(deriv_ana_loss.item())
            deriv_diff_lst.append(diff_loss.item())
            
            total_loss = ctrl_mse + self.args.loss_scale * (goal_loss + deriv_num_loss + deriv_ana_loss + diff_loss)
            total_loss_lst.append(total_loss.item())
            
        ctrl_mse = np.mean(ctrl_mse_lst)
        goal_loss = np.mean(goal_loss_lst)
        deriv_num_loss = np.mean(deriv_num_lst)
        deriv_ana_loss = np.mean(deriv_ana_lst)
        deriv_diff_loss = np.mean(deriv_diff_lst)
        total_loss = np.mean(total_loss_lst)
        
        return ctrl_mse, goal_loss, deriv_num_loss, deriv_ana_loss, deriv_diff_loss, total_loss
    
    def create_models(self):
        n_dim, n_ctrl = self.dynamic.n_dim, self.dynamic.n_control
        ctrl = MLP(n_dim, n_ctrl, **vars(self.args))
        lyapunov = NNLyapunov(
            self.dynamic, self.nominal_controller, self.args.lamb,
            nn_type = 'QuadGoalMLP',
            nn_kwargs = {
                'mlp_output_size': self.args.lyap_mlp_out,
                'hidden_size': self.args.hidden_size,
                'layer_num': self.args.layer_num,
                'activation': self.args.activation,
            },
        )
        return ctrl, lyapunov
    
    
    def get_dataloaders(self):
        dm = TrajDataModule(
            self.dynamic, self.nominal_controller, 
            **vars(self.args)
        )
        dm.setup()
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        train_loader, val_loader = self.fabric.setup_dataloaders(train_loader, val_loader)
        return train_loader, val_loader