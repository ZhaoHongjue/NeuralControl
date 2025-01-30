'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/29/2025
'''

import torch
from torch import Tensor

def discretize_AB(
    A: Tensor, 
    B: Tensor, 
    dt: float = 0.1, 
    type: str = 'forward_euler'
) -> tuple[Tensor, Tensor]:
    '''
    Reference:
    '''
    if type == 'forward_euler':
        A_d = torch.eye(A.shape[0]) + A * dt
        B_d = B * dt
    elif type == 'backward_euler':
        A_d = torch.inverse(torch.eye(A.shape[0]) - A * dt)
        B_d = A_d @ B * dt
    else:
        raise ValueError('Invalid discretization type')
    
    return A_d, B_d

def normalize(
    x: Tensor, 
    state_limits: tuple[Tensor, Tensor]
) -> Tensor:
    '''
    Reference: https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/controllers/controller_utils.py#L29
    '''
    lower, upper = state_limits
    center = (lower + upper) / 2
    range = (upper - lower) / 2
    return (x - center) / range