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
        controller: Controller = None,
        **kwargs
    ):
        super().__init__(dynamic, controller, **kwargs)
        self.P = torch.eye(dynamic.n_dim)
    
    def __call__(self, x: Tensor) -> Tensor:
        '''
        Compute the value of the Lyapunov function and its Jacobian
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): state vector
        
        Returns:
        - `v_values` (`Tensor[batch_size * 1]`): value of the Lyapunov function
        '''
        return torch.einsum('bi, ij, bj -> b', x, self.P.to(x.device), x).unsqueeze(-1)
    
    def compute_jacobian(
        self, x: Tensor, 
        create_graph: bool = False
    ) -> Tensor:
        '''
        Compute the Jacobian of the Lyapunov function
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): state vector
        '''
        return torch.einsum('bi, ij -> bj', x, self.P.to(x.device) + self.P.T.to(x.device))