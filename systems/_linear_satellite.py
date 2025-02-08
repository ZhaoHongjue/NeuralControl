'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/29/2023

Reference: https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/systems/linear_satellite.py#L11
'''

import torch, numpy as np
from torch import Tensor
from ._base_sys import CtrlAffSys

class LinearSatellite(CtrlAffSys):
    '''
    Linearized dynamics of a satellite in orbit around the Earth.
    '''
    MU          = 3.986004418e14     # Earth's gravitational parameter
    
    def __init__(self, dt: float = 0.01, **kwargs):
        super().__init__(dt = dt)
        self.a          = 500e3
        self.ux_target  = 0.0
        self.uy_target  = 0.0
        self.uz_target  = 0.0
    
    def _f(self, x: Tensor) -> Tensor:
        '''
        Linearized dynamics of a satellite in orbit around the Earth.
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): State vectors. 
        '''
        assert x.dim() == 2, 'Input must be a batch of states'
        return x @ self.A.T.to(x.device)
    
    def _g(self, x: Tensor) -> Tensor:
        '''
        Control vector [ux, uy, uz].
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): State vectors.
        '''
        assert x.dim() == 2, 'Input must be a batch of states'
        return torch.broadcast_to(self.B, (x.shape[0],) + self.B.shape).to(x.device)
    
    @property
    def A(self) -> Tensor:
        n = np.sqrt(LinearSatellite.MU / self.a ** 3)
        A_mat = torch.zeros((self.n_dim, self.n_dim))
        A_mat[[0, 1, 2], [3, 4, 5]] = 1
        A_mat[3, 0] = 3 * n ** 2
        A_mat[3, 4] = 2 * n
        A_mat[4, 3] = -2 * n
        A_mat[5, 2] = -n ** 2
        return A_mat
    
    @property
    def B(self) -> Tensor:
        B_mat = torch.zeros((self.n_dim, self.n_control))
        B_mat[[3, 4, 5], [0, 1, 2]] = 1
        return B_mat
    
    @property
    def state_limits(self) -> tuple[Tensor, Tensor]:
        upper_limit = torch.ones(self.n_dim)
        upper_limit[:3] *= 2
        lower_limit = -upper_limit
        return lower_limit, upper_limit
    
    @property
    def control_limits(self) -> tuple[Tensor, Tensor]:
        '''
        Lower and upper limits of the control inputs.
        '''
        upper_limit = torch.ones(self.n_control)
        lower_limit = -upper_limit
        return lower_limit, upper_limit
    
    @property
    def n_dim(self) -> int:
        return 6
    
    @property
    def n_control(self) -> int:
        return 3
    
    def get_mask(self, x: Tensor, type: str = 'safe') -> Tensor:
        assert x.dim() == 2, 'Input must be a batch of states'
        distance =  x[:, :3].norm(dim = 1, p = 2)
        
        if   type == 'safe'  : return torch.logical_and(distance <= 1.5, distance >= 0.25) # distance >= 0.75 # 
        elif type == 'unsafe': return torch.logical_or(distance < 0.25, distance > 1.5)
        elif type == 'goal'  : return distance < 0.25 # torch.ones(x.shape[0], dtype = torch.bool)
        else: raise ValueError('Invalid mask type')
        
    