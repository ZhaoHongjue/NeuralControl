'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/28/2025
'''
import torch
from torch import nn, Tensor
from certificates import NNLyapunov
from torch.nn import functional as F

class LyapNNLoss:
    def __init__(
        self, 
        lyapunov: NNLyapunov, 
        lamb: float = 1.0,
        dt: float = 0.01,
        device: torch.device = torch.device('cpu'),
    ):
        self.lyapunov = lyapunov
        self.lamb = lamb
        self.dt = dt
        self.device = device
        self.goal_point = self.lyapunov.dynamic.goal_point.to(device)  
        
    def __call__(
        self, 
        curr_xs: Tensor, 
        next_xs: Tensor,
        **kwargs
    ) -> Tensor:
        goal_loss = self.goal_loss()
        deriv_num_loss, deriv_ana_loss, diff_loss = self.deriv_loss(curr_xs, next_xs)
        return goal_loss, deriv_num_loss, deriv_ana_loss, diff_loss
        
    def goal_loss(self) -> Tensor:
        return self.lyapunov(self.goal_point).mean()
    
    def deriv_loss(self, curr_xs: Tensor, next_xs: Tensor) -> Tensor:
        # Lyapunov Numerical loss - Compute the Lyapunov function and its derivative
        curr_vs = self.lyapunov(curr_xs)
        next_vs = self.lyapunov(next_xs)
        dot_vs_num = (next_vs - curr_vs) / self.dt
        deriv_num_loss = F.relu(dot_vs_num + self.lamb * curr_vs).mean()
        
        # Lyapunov Analytical loss - Compute the Lyapunov function and its derivative
        grad_vs = self.lyapunov.compute_jacobian(curr_xs)
        dot_xs = (next_xs - curr_xs) / self.dt
        dot_vs_ana = (grad_vs * dot_xs).sum(dim = -1, keepdim = True)
        deriv_ana_loss = F.relu(dot_vs_ana + self.lamb * curr_vs).mean()
        
        # Diff Loss
        diff_loss = F.mse_loss(dot_vs_ana, dot_vs_num).mean()
        
        return deriv_num_loss, deriv_ana_loss, diff_loss
