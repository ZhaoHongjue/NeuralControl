'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/29/2023

Reference: https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/systems/linear_satellite.py#L11
'''

import torch
from torch import Tensor

from ._base_sys import CtrlAffSys

class InvertedPendulum(CtrlAffSys):
    def __init__(self, dt: float = 0.01):
        self.dt = dt
        self.m = 1.0
        self.L = 1.0
        self.b = 0.01
        self.g = 9.81
    
    def _f(self, x: Tensor) -> Tensor:
        assert x.dim() == 2, 'Input must be a batch of states'
        th, th_dot = x.T
        dth = th_dot
        dth_dot = (self.g / self.L) * torch.sin(th) - (self.b / (self.m * self.L**2)) * th_dot
        return torch.stack([dth, dth_dot], dim = 1)
    
    def _g(self, x: Tensor) -> Tensor:
        assert x.dim() == 2, 'Input must be a batch of states'
        return torch.broadcast_to(self.B, (x.shape[0],) + self.B.shape).to(x.device)
    
    @property
    def B(self) -> Tensor:
        return torch.tensor([[0.0], [1.0 / (self.m * self.L**2)]])
    
    @property
    def state_limits(self) -> tuple[Tensor, Tensor]:
        upper_limit = torch.pi * torch.ones(self.n_dim)
        lower_limit = -upper_limit
        return lower_limit, upper_limit
    
    @property
    def control_limits(self) -> tuple[Tensor, Tensor]:
        return torch.tensor([-1000.0]), torch.tensor([1000.0])
    
    @property
    def n_dim(self) -> int:
        return 2
    
    @property
    def n_control(self) -> int:
        return 1
    
    def get_mask(self, x: Tensor, type: str = 'safe') -> Tensor:
        if type == 'safe': return x.norm(dim = -1) <= 0.5
        elif type == 'unsafe': return x.norm(dim = -1) > 1.5
        elif type == 'goal': return x.norm(dim = -1) <= 0.3
        else: raise ValueError(f"Invalid mask type: {type}")