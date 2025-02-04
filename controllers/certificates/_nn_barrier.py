'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/30/2025
'''

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from ._nn_certificate import NNCertificate

class NNBarrier(NNCertificate):
    @property
    def certif_type(self) -> str:
        return 'barrier'
    
    def compute_loss(
        self, 
        xs: Tensor,
        goal_mask: Tensor,
        safe_mask: Tensor,  
        unsafe_mask: Tensor,
    ) -> Tensor:
        safe_violation_loss, unsafe_violation_loss = self.compute_violation_loss(xs, goal_mask, safe_mask, unsafe_mask)
        relaxation_loss = self.compute_relaxation_loss(xs)
        return safe_violation_loss + unsafe_violation_loss + relaxation_loss 
    
    def compute_violation_loss(
        self, 
        xs: Tensor,
        goal_mask: Tensor,
        safe_mask: Tensor,  
        unsafe_mask: Tensor,
    ) -> Tensor:
        '''
        Compute the loss for the NN Barrier function.
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): state vector
        - `goal_mask` (`Tensor[batch_size]`): mask for goal region
        - `safe_mask` (`Tensor[batch_size]`): mask for safe region
        - `unsafe_mask` (`Tensor[batch_size]`): mask for unsafe region
        
        Returns:
        - `loss` (`Tensor[1]`): loss value
        '''
        eps = 1e-2
        v_values = self(xs)
        
        safe_violation_loss = 100 * F.relu(eps - v_values[safe_mask]).mean()
        unsafe_violation_loss = 100 * F.relu(eps + v_values[unsafe_mask]).mean()
        
        return safe_violation_loss, unsafe_violation_loss
        
    def compute_relaxation_loss(self, xs: Tensor) -> Tensor:
        return (self.get_relaxation(xs)**2).mean()
