'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/29/2025
'''

import torch
from torch import Tensor
from ._base_controller import Controller
from systems import CtrlAffSys


class RandomController(Controller):
    '''
    Random controller
    '''
    def __init__(
        self, 
        dynamic: CtrlAffSys,
        dt: float = 0.01, 
        **kwargs
    ):
        super().__init__(dynamic, dt)
        
    def __call__(self, x: Tensor) -> Tensor:
        '''
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): State vectors.
        
        Return:
        - `Tensor[batch_size * N_CONTROL]`: Control vectors.
        '''
        return torch.rand(x.shape[0], self.dynamic.n_control).to(x.device)