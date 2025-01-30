'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/29/2025
'''

import torch
from torch import Tensor

from systems import CtrlAffSys
from ._base_controller import Controller
from .functional import continuous_lqr, discrete_lqr, discretize_AB


class LQRController(Controller):
    def __init__(
        self, 
        dynamic: CtrlAffSys,
        dt: float = 0.1, 
        Q: Tensor = None,
        R: Tensor = None,
        type: str = 'discrete',
        **kwargs
    ):
        super().__init__(dynamic, dt)
        # Assign Q
        if Q is not None: 
            assert Q.dim() == 2, 'Q must be a 2D tensor'
            self.Q = Q
        else:
            self.Q = torch.eye(dynamic.n_dim)
            
        # Assign R
        if R is not None:
            assert R.dim() == 2, 'R must be a 2D tensor'
            self.R = R
        else:
            self.R = torch.eye(dynamic.n_control)
            
        # Assign K
        if type == 'discrete':
            self.K = discrete_lqr(self.dynamic.A, self.dynamic.B, self.Q, self.R)
        elif type == 'continuous':
            A_d, B_d = discretize_AB(self.dynamic.A, self.dynamic.B, self.dynamic.dt)
            self.K = continuous_lqr(self.dynamic.A, self.dynamic.B, self.Q, self.R)
        else:
            raise ValueError('Type must be either discrete or continuous')
        
    def __call__(self, x: Tensor) -> Tensor:
        '''
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): State vectors.
        
        Return:
        - `Tensor[batch_size * N_CONTROL]`: Control vectors.
        '''
        return -x @ self.K.T