'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/02/2025
'''

import torch
from torch import nn, Tensor

class MLP(nn.Module):
    def __init__(
        self, 
        input_size: int,
        output_size: int = 1,
        hidden_size: int = 32,
        layer_num: int = 2,
        activation: str = 'ReLU',
        **kwargs,
    ) -> None:
        super().__init__()
        self.l0 = nn.Linear(input_size, hidden_size)
        self.activation0 = eval(f'nn.{activation}')()
        
        for i in range(layer_num):
            if i == layer_num - 1:
                setattr(self, f'linear{i+1}', nn.Linear(hidden_size, output_size))
            else:
                setattr(self, f'linear{i+1}', nn.Linear(hidden_size, hidden_size))
                setattr(self, f'activation{i+1}', eval(f'nn.{activation}')())
                
    def forward(self, x: Tensor) -> Tensor:
        for module in self.children():
            x = module(x)
        return x