'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   03/03/2025
'''
import torch
from torch import nn, Tensor
from certificates import NNBarrier
from torch.nn import functional as F

class BarrierNNLoss:
    def __init__(
        self, 
        barrier: NNBarrier, 
        lamb: float = 1.0,
        dt: float = 0.01,
        device: torch.device = torch.device('cpu'),
    ):
        self.barrier = barrier
        self.dynamic = barrier.dynamic
        self.lamb = lamb
        self.dt = dt
        self.device = device
        
    def __call__(
        self, 
        curr_xs: Tensor, 
        next_xs: Tensor,
        **kwargs
    ) -> Tensor:
        boundary_loss = self.boundary_loss(curr_xs)
        deriv_num_loss, deriv_ana_loss, diff_loss = self.deriv_loss(curr_xs, next_xs)
        return boundary_loss, deriv_num_loss, deriv_ana_loss, diff_loss
        
    def boundary_loss(self, xs: Tensor) -> Tensor:
        safe_flags = self.dynamic.get_safe_flags(xs)
        hs = self.barrier(xs)
        return F.relu(-1 * safe_flags * hs).mean()
        
    def deriv_loss(self, curr_xs: Tensor, next_xs: Tensor) -> Tensor:
        # Barrier Numerical loss - Compute the Barrier function and its derivative
        curr_hs = self.barrier(curr_xs)
        next_hs = self.barrier(next_xs)
        dot_hs_num = (next_hs - curr_hs) / self.dt
        deriv_num_loss = F.relu(dot_hs_num + self.lamb * curr_hs).mean()
        
        # Barrier Analytical loss - Compute the Barrier function and its derivative
        grad_hs = self.barrier.compute_jacobian(curr_xs)
        dot_xs = (next_xs - curr_xs) / self.dt
        dot_hs_ana = (grad_hs * dot_xs).sum(dim = -1, keepdim = True)
        deriv_ana_loss = F.relu(dot_hs_ana + self.lamb * curr_hs).mean()
        
        # Diff Loss
        diff_loss = F.mse_loss(dot_hs_ana, dot_hs_num).mean()
        
        return deriv_num_loss, deriv_ana_loss, diff_loss
