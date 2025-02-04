'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/30/2025
'''

import torch
from torch import Tensor

from systems import CtrlAffSys
from .. import Controller
from ._base import Certificate


class QuadBarrier(Certificate):
    '''
    Only used for testing purposes.
    '''
    def __init__(
        self, 
        dynamic: CtrlAffSys, 
        nominal_controller: Controller = None,
        lamb: float = 1.0,
        r_penalty: float = 1.0,
        **kwargs
    ):
        super().__init__(dynamic, nominal_controller, lamb, r_penalty, **kwargs)
        self.P = torch.eye(dynamic.n_dim)
    
    @property
    def certif_type(self) -> str:
        return 'barrier'
    
    def _value(self, x: Tensor) -> Tensor:
        return x @ self.P.to(x.device) @ x
    
    def compute_jacobian(self, xs: Tensor) -> Tensor:
        '''
        Compute the Jacobian of the Lyapunov function
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): state vector
        '''
        return torch.einsum('bi, ij -> bj', xs, self.P.to(xs.device) + self.P.T.to(xs.device))