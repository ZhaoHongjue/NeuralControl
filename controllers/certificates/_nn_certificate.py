'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/02/2025
'''

from abc import abstractmethod

import torch
from torch import nn, Tensor

from systems import CtrlAffSys
from models import *
from .. import Controller
from ._base import Certificate


class NNCertificate(nn.Module, Certificate):
    def __init__(
        self,
        dynamic: CtrlAffSys,
        controller: Controller = None,
        nn_type: str = 'MLP',
        nn_kwargs: dict = {
            'hidden_size': 32,
            'layer_num': 2,
        },
        **kwargs,
    ) -> None:
        Certificate.__init__(self, dynamic, controller)
        nn.Module.__init__(self)
        if nn_type == 'MLP':
            self.dnn = MLP(dynamic.n_dim, **nn_kwargs)
        else:
            raise ValueError(f'NN type {nn_type} not supported')
    
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
        