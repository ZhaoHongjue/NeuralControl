import os, argparse, logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

import torch, numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import functional as F
from lightning.fabric import Fabric

from systems import LinearSatellite
from controllers import Controller
from certificates import NNBarrier
from models import MLP, LimitMLP
from data import TrajDataModule
import utils
from utils.loss import BarrierNNLoss


class LinSate_CBF_Trainer:
    def __init__(
        self,
        dynamic: LinearSatellite,
        nominal_controller: Controller,
        args: argparse.Namespace,
        ckpt_pth: str = './checkpoints',
    ):
        utils.info_args_table(args)
        self.dynamic, self.nominal_controller = dynamic, nominal_controller
        self.args = args
        
        self.ckpt_pth = ckpt_pth
        self.barrier_ckpt_pth = f'{ckpt_pth}/lin_sate_barrier-lamb{args.lamb}/{args.ctrl_type}'
        self.ctrl_ckpt_pth = f'{ckpt_pth}/lin_sate_ctrl-lamb{args.lamb}/{args.ctrl_type}'
        os.makedirs(self.barrier_ckpt_pth, exist_ok = True)
        os.makedirs(self.ctrl_ckpt_pth, exist_ok = True)
        
        self.fabric = Fabric(accelerator = 'cuda', devices = [args.cuda,],)
        device = torch.device(f'cuda:{args.cuda}')
        
        self.nn_ctrl, self.barrier = self.create_models()
        self.ctrl_mat = self.dynamic.B.to(device) # (n_dim, n_control)
        
        self.ctrl_optim = Adam(self.nn_ctrl.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        self.ctrl_scheduler = CosineAnnealingLR(self.ctrl_optim, args.n_epochs, eta_min = 0.1 * args.lr)
        
        self.barrier_optim = Adam(self.barrier.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        self.barrier_scheduler = CosineAnnealingLR(self.barrier_optim, args.n_epochs, eta_min = 0.1 * args.lr)
        
        self.nn_ctrl, self.ctrl_optim = self.fabric.setup(self.nn_ctrl, self.ctrl_optim)  
        self.barrier, self.barrier_optim = self.fabric.setup(self.barrier, self.barrier_optim)
        
        self.train_loader, self.val_loader = self.get_dataloaders()
        self.loss_fn = BarrierNNLoss(self.barrier, self.args.lamb, self.args.dt, device)
        
        self.nn_ctrl: MLP
        self.barrier: NNBarrier
    
    
    def train(self):
        best_epoch, min_val_loss = 0, float('inf')
        for epoch in range(self.args.n_epochs):
            train_ctrl_mse, train_boundary_loss, train_deriv_num_loss, train_deriv_ana_loss, train_deriv_diff_loss, train_total_loss = self.train_step()
            val_ctrl_mse, val_boundary_loss, val_deriv_num_loss, val_deriv_ana_loss, val_deriv_diff_loss, val_total_loss = self.validate_step()
            
            if val_total_loss < min_val_loss:
                min_val_loss, best_epoch = val_total_loss, epoch
                utils.save_checkpoint(self.nn_ctrl, self.ctrl_optim, epoch, f'{self.ctrl_ckpt_pth}/best_ctrl.pt')
                utils.save_checkpoint(self.barrier, self.barrier_optim, epoch, f'{self.barrier_ckpt_pth}/best_barrier.pt')
            
            logging.info(f'Epoch {epoch:4d} - Train: {train_total_loss:.4e} | MSE: {train_ctrl_mse:.4e} | Boundary: {train_boundary_loss:.4e} | Deriv Num: {train_deriv_num_loss:.4e} | Deriv Ana: {train_deriv_ana_loss:.4e} | Deriv Diff: {train_deriv_diff_loss:.4e} | Best Epoch: {best_epoch:4d}')
            logging.info(f'Epoch {epoch:4d} - Val  : {val_total_loss:.4e} | MSE: {val_ctrl_mse:.4e} | Boundary: {val_boundary_loss:.4e} | Deriv Num: {val_deriv_num_loss:.4e} | Deriv Ana: {val_deriv_ana_loss:.4e} | Deriv Diff: {val_deriv_diff_loss:.4e} | Best Epoch: {best_epoch:4d}')
            logging.info('-' * 150)
            
            if epoch % 100 == 0:
                torch.save(self.nn_ctrl.state_dict(), f'{self.ctrl_ckpt_pth}/epoch{epoch}.pt')
                torch.save(self.barrier.state_dict(), f'{self.barrier_ckpt_pth}/epoch{epoch}.pt')
        
        
    def train_step(self):
        total_loss_lst, ctrl_mse_lst, boundary_loss_lst = [], [], []
        deriv_diff_lst, deriv_num_lst, deriv_ana_lst = [], [], []
        for curr_xs, curr_us, next_xs in self.train_loader:
            # Control loss - Restrict the control to the nominal controller
            pred_delta_us = self.nn_ctrl(curr_xs)
            ctrl_mse = torch.linalg.norm(pred_delta_us, ord = 2, dim = -1).mean()
            ctrl_mse_lst.append(ctrl_mse.item())
            
            pred_us = curr_us + pred_delta_us
            pred_next_xs = next_xs + self.args.dt * pred_us @ self.ctrl_mat.T.to(pred_us.device)
            boundary_loss, deriv_num_loss, deriv_ana_loss, diff_loss = self.loss_fn(curr_xs, pred_next_xs)
            boundary_loss_lst.append(boundary_loss.item())
            deriv_num_lst.append(deriv_num_loss.item())
            deriv_ana_lst.append(deriv_ana_loss.item())
            deriv_diff_lst.append(diff_loss.item())
            
            total_loss = ctrl_mse + self.args.loss_scale * (boundary_loss + deriv_num_loss + deriv_ana_loss + diff_loss)
            total_loss_lst.append(total_loss.item())
            
            self.barrier_optim.zero_grad()
            self.ctrl_optim.zero_grad()
            self.fabric.backward(total_loss)
            self.barrier_optim.step()
            self.ctrl_optim.step()
            self.ctrl_scheduler.step()
            self.barrier_scheduler.step()
        
        ctrl_mse = np.mean(ctrl_mse_lst)
        boundary_loss = np.mean(boundary_loss_lst)
        deriv_num_loss = np.mean(deriv_num_lst)
        deriv_ana_loss = np.mean(deriv_ana_lst)
        deriv_diff_loss = np.mean(deriv_diff_lst)
        total_loss = np.mean(total_loss_lst)
        
        return ctrl_mse, boundary_loss, deriv_num_loss, deriv_ana_loss, deriv_diff_loss, total_loss
    
    
    @torch.no_grad()
    def validate_step(self):
        total_loss_lst, ctrl_mse_lst, boundary_loss_lst = [], [], []
        deriv_diff_lst, deriv_num_lst, deriv_ana_lst = [], [], []
        for curr_xs, curr_us, next_xs in self.train_loader:
            # Control loss - Restrict the control to the nominal controller
            pred_delta_us = self.nn_ctrl(curr_xs)
            ctrl_mse = torch.linalg.norm(pred_delta_us, ord = 2, dim = -1).mean()
            ctrl_mse_lst.append(ctrl_mse.item())
            
            pred_us = curr_us + pred_delta_us
            pred_next_xs = next_xs + self.args.dt * pred_us @ self.ctrl_mat.T.to(pred_us.device)
            boundary_loss, deriv_num_loss, deriv_ana_loss, diff_loss = self.loss_fn(curr_xs, pred_next_xs)
            boundary_loss_lst.append(boundary_loss.item())
            deriv_num_lst.append(deriv_num_loss.item())
            deriv_ana_lst.append(deriv_ana_loss.item())
            deriv_diff_lst.append(diff_loss.item())
            
            total_loss = ctrl_mse + self.args.loss_scale * (boundary_loss + deriv_num_loss + deriv_ana_loss + diff_loss)
            total_loss_lst.append(total_loss.item())
            
        ctrl_mse = np.mean(ctrl_mse_lst)
        boundary_loss = np.mean(boundary_loss_lst)
        deriv_num_loss = np.mean(deriv_num_lst)
        deriv_ana_loss = np.mean(deriv_ana_lst)
        deriv_diff_loss = np.mean(deriv_diff_lst)
        total_loss = np.mean(total_loss_lst)
        
        return ctrl_mse, boundary_loss, deriv_num_loss, deriv_ana_loss, deriv_diff_loss, total_loss
    
    
    def create_models(self):
        n_dim, n_ctrl = self.dynamic.n_dim, self.dynamic.n_control
        if self.args.ctrl_type == 'LimitMLP':
            l_lim, u_lim = self.dynamic.control_limits
            # ctrl = LimitMLP(n_dim, n_dim, l_lim, u_lim, **vars(self.args))
            ctrl = LimitMLP(n_dim, n_ctrl, l_lim, u_lim, **vars(self.args))
        elif self.args.ctrl_type == 'MLP':
            # ctrl = MLP(n_dim, n_dim, **vars(self.args))
            ctrl = MLP(n_dim, n_ctrl, **vars(self.args))
        else:
            raise ValueError(f'Invalid controller type: {self.args.ctrl_type}')
        
        barrier = NNBarrier(
            self.dynamic, self.nominal_controller, self.args.lamb,
            nn_type = 'MLP', nn_kwargs = vars(self.args)
        )
        
        return ctrl, barrier
    
    
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