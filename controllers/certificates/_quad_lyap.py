'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/30/2025

Reference: https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/controllers/clf_controller.py
'''

from warnings import catch_warnings

import torch
from torch import Tensor

from systems import CtrlAffSys
from .. import Controller
from .. import functional as F
from ._base import Certificate


class QuadLyapunov(Certificate):
    def __init__(
        self, 
        dynamic: CtrlAffSys, 
        controller: Controller = None,
        P: Tensor = None,
        **kwargs
    ):
        super().__init__(dynamic, controller, **kwargs)
        if P is None:
            with catch_warnings(record = True) as w:
                self.P = F.compute_sys_lyapunov_p(dynamic, controller)
                if w: 
                    print('Cannot compute the Lyapunov matrix, using the identity matrix instead')
                    self.P = torch.eye(dynamic.n_dim)
        else:
            assert P.shape == (dynamic.n_dim, dynamic.n_dim), 'P must be a square matrix'
            self.P = P
    
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