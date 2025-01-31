'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/30/2025

Reference: https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/controllers/clf_controller.py
'''

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.autograd.functional import jacobian

from systems import CtrlAffSys
from .. import Controller, ConstantController


class Certificate(ABC):
    def __init__(
        self,
        dynamic: CtrlAffSys,
        controller: Controller = None,
        **kwargs,
    ) -> None:
        self.dynamic = dynamic
        if controller is None:
            self.controller = ConstantController(dynamic, dt = dynamic.dt)
        else:
            self.controller = controller
    
    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def compute_jacobian(
        self, 
        x: Tensor, 
        create_graph: bool = False
    ) -> Tensor:
        '''
        Compute the Jacobian of the Lyapunov function
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): state vector
        '''
        return torch.cat([
            jacobian(self, x_i, create_graph = create_graph).reshape(1, -1)
            for x_i in x.unsqueeze(1)
        ], dim = 0)
        
    def compute_lie_deriv(self, x: Tensor) -> Tensor:
        '''
        Compute the Lie derivative of the Lyapunov function
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): state vector
        '''
        f, g = self.dynamic.get_f_ang_g(x)
        v_jacob = self.compute_jacobian(x)
        Lf_v = torch.einsum('bi, bi -> b', f, v_jacob).unsqueeze(-1)
        Lg_v = torch.einsum('bij, bi -> bj', g, v_jacob)
        return Lf_v, Lg_v
        