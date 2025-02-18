'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/17/2025
'''

import torch
from torch import nn, Tensor
from ._mlp import MLP


class QuadNN(nn.Module):
    def __init__(
        self, 
        input_size: int,
        mlp_output_size: int = 32,
        hidden_size: int = 32,
        layer_num: int = 2,
        activation: str = 'ReLU',
        **kwargs,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_size, mlp_output_size, hidden_size, 
            layer_num, activation, **kwargs
        )
        
    def forward(self, x: Tensor) -> Tensor:
        mlp_out = self.mlp(x)
        return (mlp_out * mlp_out).sum(dim = -1, keepdim = True)
    
    