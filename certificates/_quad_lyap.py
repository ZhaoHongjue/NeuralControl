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
from controllers import Controller
from utils import compute_sys_lyapunov_p
from ._base import Certificate


class QuadLyapunov(Certificate):
    def __init__(
        self, 
        dynamic: CtrlAffSys, 
        nominal_controller: Controller = None,
        lamb: float = 1.0,
        r_penalty: float = 1.0,
        P: Tensor = None,
        **kwargs
    ):
        super().__init__(dynamic, nominal_controller, lamb, r_penalty, **kwargs)
        if P is None:
            with catch_warnings(record = True) as w:
                self.P = compute_sys_lyapunov_p(dynamic, nominal_controller)
                if w: 
                    print('Cannot compute the Lyapunov matrix, using the identity matrix instead')
                    self.P = torch.eye(dynamic.n_dim)
        else:
            assert P.shape == (dynamic.n_dim, dynamic.n_dim), 'P must be a square matrix'
            self.P = P
            
    @property
    def certif_type(self) -> str:
        return 'lyapunov'
            
    def _value(self, x: Tensor) -> Tensor:
        return x @ self.P.to(x.device) @ x
    
    def compute_jacobian(self, xs: Tensor) -> Tensor:
        '''
        Compute the Jacobian of the Lyapunov function
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): state vector
        '''
        return torch.einsum('bi, ij -> bj', xs, self.P.to(xs.device) + self.P.T.to(xs.device))