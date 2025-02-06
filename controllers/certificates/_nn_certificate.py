'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/02/2025
'''

from abc import abstractmethod

import torch
from torch import nn, Tensor

from systems import CtrlAffSys
from utils import init_nn_weights
from models import *
from .. import Controller
from ._base import Certificate


class NNCertificate(nn.Module, Certificate):
    def __init__(
        self,
        dynamic: CtrlAffSys,
        nominal_controller: Controller = None,
        lamb: float = 1.0,
        r_penalty: float = 1.0,
        nn_type: str = 'MLP',
        nn_kwargs: dict = {
            'hidden_size': 32,
            'layer_num': 2,
            'activation': 'ReLU',
        },
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)
        Certificate.__init__(self, dynamic, nominal_controller, lamb, r_penalty, **kwargs)
        if nn_type == 'MLP':
            self.dnn = MLP(dynamic.n_dim, **nn_kwargs)
            init_nn_weights(self.dnn)
        else:
            raise ValueError(f'NN type {nn_type} not supported')
        self.qp_solver = self.init_qp_solver()
    
    def forward(self, xs: Tensor) -> Tensor:
        return self.dnn(xs)
    
    @property
    def certif_type(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def compute_loss(self, xs: Tensor) -> Tensor:
        '''
        Compute the loss function
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): state vector
        '''
        raise NotImplementedError
    
    def _value(self, x: Tensor) -> Tensor:
        '''
        Compute the value of the Lyapunov function
        
        Args:
        - `x` (`Tensor[N_DIM]`): state vector
        
        Returns:
        - `v` (`Tensor[1]`): value of the Lyapunov function
        '''
        return self.dnn(x).squeeze(-1)