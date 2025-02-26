'''
Author:  Hongjue Zhao
Email:   hongjue2@illinois.edu
Date:    02/27/2025
'''

import os, pandas as pd

import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, random_split
from lightning import LightningDataModule

from systems import CtrlAffSys
from controllers import Controller, ConstantController
from utils import simulate_con_system


class TrajDataModule(LightningDataModule):
    def __init__(
        self,
        dynamic: CtrlAffSys,
        controller: Controller,
        batch_size: int = 1000,
        n_traj: int = 100,
        T: float = 10.,
        dt: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.dynamic, self.controller = dynamic, controller
        self.batch_size = batch_size
        self.n_traj = n_traj
        self.T, self.dt = T, dt
        

    def setup(self, stage: str = 'fit'):
        batch_x0 = self.dynamic.sample_state_space(self.n_traj)
        _, batch_xs = simulate_con_system(
            self.dynamic, self.controller, batch_x0, self.T, self.dt
        )
        curr_xs, curr_us, next_xs = self.get_batch_traj_data(batch_xs)
        dataset = TensorDataset(curr_xs, curr_us, next_xs)
        self.train_dataset, self.test_dataset = random_split(dataset, [0.8, 0.2])
        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
    
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle = False)
    
    
    def get_single_traj_data(self, xs):
        prev_xs = xs[:-1]
        next_xs = xs[1:]
        prev_us = self.controller(prev_xs)
        single_data = torch.cat([prev_xs, prev_us, next_xs], dim=1)
        return single_data


    def get_batch_traj_data(self, batch_xs):
        all_data = torch.vmap(self.get_single_traj_data)(batch_xs)
        all_data = all_data.reshape(-1, all_data.shape[-1])
        n_dim, n_control = self.dynamic.n_dim, self.dynamic.n_control
        curr_xs, curr_us, next_xs = torch.split(
            all_data, [n_dim, n_control, n_dim], dim = -1
        )
        return curr_xs, curr_us, next_xs