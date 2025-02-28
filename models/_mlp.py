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
    
    
class LimitMLP(nn.Module):
    def __init__(
        self, 
        input_size: int,
        output_size: int,
        lower_limit: Tensor,
        upper_limit: Tensor,
        hidden_size: int = 32,
        layer_num: int = 2,
        activation: str = 'ReLU',
        **kwargs,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            input_size, output_size, hidden_size, 
            layer_num, activation, **kwargs
        )
        self.lower_limit = nn.Parameter(lower_limit, requires_grad = False)
        self.upper_limit = nn.Parameter(upper_limit, requires_grad = False)
        
    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(self.mlp(x), self.lower_limit, self.upper_limit)
    
    
class QuadMLP(nn.Module):
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
    

class QuadGoalMLP(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        goal_point: Tensor = None,
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
        if goal_point is None:
            goal_point = torch.zeros(input_size)
        self.goal_point = nn.Parameter(goal_point.reshape(input_size,), requires_grad = False)
        self.linear = nn.Linear(input_size, mlp_output_size, bias = False)
        
    def forward(self, x: Tensor) -> Tensor:
        mlp_out = self.mlp(x)
        goal_out = self.linear(x - self.goal_point)
        term1 = (mlp_out * mlp_out).sum(dim = -1, keepdim = True)
        term2 = (goal_out * goal_out).sum(dim = -1, keepdim = True)
        return term1 + term2