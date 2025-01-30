'''
Author:     Hongjue Zhao
Email:      hongjue2@illinois.edu
Date:       01/29/2025
'''

import torch
from torch import Tensor
from ._base_controller import Controller
from systems import CtrlAffSys


class ConstantController(Controller):
    '''
    Constant controller
    '''
    def __init__(
        self, 
        dynamic: CtrlAffSys,
        dt: float = 0.1, 
        u: Tensor = None,
        **kwargs
    ):
        super().__init__(dynamic, dt)
        if u is not None: 
            assert u.dim() == 1, 'Control input must be a 1D tensor'
            self.u = u
        else: 
            self.u = torch.zeros(dynamic.n_control)
        
    def __call__(self, x: Tensor) -> Tensor:
        '''
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): State vectors.
        
        Return:
        - `Tensor[batch_size * N_CONTROL]`: Control vectors.
        '''
        return torch.broadcast_to(self.u, (x.shape[0],) + self.u.shape).to(x.device)