'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/05/2025
'''

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from controllers.certificates import NNCertificate, NNBarrier, NNLyapunov


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
    

def barrier_deriv_loss(
    nn_barrier: NNBarrier,
    xs: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    '''
    Reference:
    '''
    us = nn_barrier.nominal_controller(xs)
    hs = nn_barrier(xs)
    Lf_h, Lg_h = nn_barrier.compute_lie_deriv(xs)
    dh = Lf_h + torch.einsum('ij,ij->i', Lg_h, us).unsqueeze(1)
    deriv_loss = F.relu(-dh - nn_barrier.lamb * hs)
    if reduction == 'mean':  return deriv_loss.mean()
    elif reduction == 'sum': return deriv_loss.sum()
    else: raise ValueError(f'Unknown reduction method: {reduction}')
    

def lyap_deriv_loss(
    nn_lyap: NNLyapunov,
    xs: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    '''
    Reference: https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/controllers/neural_cbf_controller.py
    
    Implement the loss function for the Lyapunov function in Safe Control with learned certificates.
    
    Args:
    - `nn_lyap` (`NNLyapunov`): Lyapunov function
    - `xs` (`Tensor[batch_size, n_dim]`): state
    - `reduction` (`str`): reduction method
    
    Returns:
    - `loss` (`Tensor`): loss
    '''
    Lf_V, Lg_V = nn_lyap.compute_lie_deriv(xs)
    us = nn_lyap.nominal_controller(xs)
    dV = Lf_V + torch.einsum('ij,ij->i', Lg_V, us).unsqueeze(1)
    deriv_loss = F.relu(dV)
    if reduction == 'mean':  return deriv_loss.mean()
    elif reduction == 'sum': return deriv_loss.sum()
    else: raise ValueError(f'Unknown reduction method: {reduction}')

def lyap_qp_loss(
    nn_lyap: NNLyapunov,
    xs: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    Lf_V, Lg_V = nn_lyap.compute_lie_deriv(xs)
    us, rs = nn_lyap._solve_qp_cvxplayers(xs)
    dV = Lf_V + torch.einsum('ij,ij->i', Lg_V, us).unsqueeze(1)
    V = nn_lyap(xs)
    deriv_loss = F.relu(dV + nn_lyap.lamb * V)
    relax_loss = F.relu(rs)
    if reduction == 'mean':  return deriv_loss.mean(), relax_loss.mean()
    elif reduction == 'sum': return deriv_loss.sum(), relax_loss.sum()
    else: raise ValueError(f'Unknown reduction method: {reduction}')

def certif_relaxation_loss(
    nn_certif: NNCertificate,
    xs: Tensor,
    reduction: str = 'mean'
) -> Tensor:
    '''
    Reference: https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/controllers/neural_cbf_controller.py
    
    Implement the loss function for the barrier boundary in Safe Control with learned certificates.
    '''
    relaxation = F.relu(nn_certif.get_relaxation(xs))
    if reduction == 'mean':  return relaxation.mean()
    elif reduction == 'sum': return relaxation.sum()
    else: raise ValueError(f'Unknown reduction method: {reduction}')
