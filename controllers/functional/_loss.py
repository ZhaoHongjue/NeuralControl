'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/05/2025
'''

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from controllers.certificates import NNCertificate, NNBarrier


def barrier_boundary_loss(
    hs: Tensor, 
    safe_mask: Tensor, 
    unsafe_mask: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    '''
    Reference: https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/controllers/neural_cbf_controller.py
    
    Implement the loss function for the barrier boundary in Safe Control with learned certificates.
    
    Args:
    - `hs` (`Tensor[batch_size]`): barrier values
    - `safe_mask` (`Tensor[batch_size]`): mask for safe region
    - `unsafe_mask` (`Tensor[batch_size]`): mask for unsafe region
    '''
    safe_loss = F.relu(-hs[safe_mask])
    unsafe_loss = F.relu(hs[unsafe_mask])
    
    if reduction == 'mean':
        return safe_loss.mean(), unsafe_loss.mean()
    elif reduction == 'sum':
        return safe_loss.sum(), unsafe_loss.sum()
    else:
        raise ValueError(f'Unknown reduction method: {reduction}')
    

def barrier_relaxation_loss(
    nn_barrier: NNBarrier,
    xs: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    '''
    Reference: https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/controllers/neural_cbf_controller.py
    
    Implement the loss function for the barrier boundary in Safe Control with learned certificates.
    '''
    relaxation = F.relu(nn_barrier.get_relaxation(xs))
    if reduction == 'mean':  return relaxation.mean()
    elif reduction == 'sum': return relaxation.sum()
    else: raise ValueError(f'Unknown reduction method: {reduction}')