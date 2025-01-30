'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/29/2025

Reference: https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/systems/control_affine_system.py#L23
'''

from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.autograd.functional import jacobian


class CtrlAffSys(ABC):
    '''
    Base class for control
    '''
    def __init__(self, dt: float = 0.01, **kwargs,):
        super().__init__()
        self.dt = dt
    
    @abstractmethod
    def _f(self, x: Tensor) -> Tensor:
        '''
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): State vectors. 
        
        Return:
        - `Tensor[batch_size * N_DIM]`: f(x).
        '''
        raise NotImplementedError
    
    @abstractmethod
    def _g(self, x: Tensor) -> Tensor:
        '''
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): State vectors.
        
        Return:
        - `Tensor[batch_size * N_DIM * N_CONTROL]`: Control matrix.
        '''
        raise NotImplementedError
    
    def get_f_ang_g(self, x: Tensor) -> tuple[Tensor, Tensor]:
        '''
        Get the dynamics of the system and the control matrix.
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]` or `Tensor[batch_size * t_step * N_DIM]`): state vector
        
        Return:
        - `Tuple[Tensor[batch_size * N_DIM], Tensor[batch_size * N_DIM * N_CONTROL]]`: f(x), g(x).
        '''
        if x.dim() == 2:
            f, g = self._f(x), self._g(x)
        elif x.dim() == 3:
            f = torch.stack([self._f(xi) for xi in x]).to(x.device)
            g = torch.stack([self._g(xi) for xi in x]).to(x.device)
        else:
            raise ValueError('Invalid input dimension')
        return f, g
    
    def closed_loop_dynamics(self, x: Tensor, u: Tensor) -> Tensor:
        '''
        Closed-loop dynamics of a satellite in orbit around the Earth.
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): State vectors.
        - `u` (`Tensor[batch_size * N_CONTROL]`): Control vectors.
        '''
        assert x.dim() == 2, 'Input must be a batch of states'
        assert u.dim() == 2, 'Input must be a batch of controls'
        return self._f(x) + torch.einsum('bsc,bc->bs', self._g(x), u)
    
    @property
    def A(self):
        '''
        Compute the linearized A matrix.
        '''
        return jacobian(
            lambda x: self.closed_loop_dynamics(x, self.u_eq), 
            self.goal_point
        ).squeeze(0, 2)
        
    @property
    def B(self):
        B_mat = self._g(self.goal_point).squeeze(0)
        return B_mat
    
    @property
    def goal_point(self) -> Tensor:
        return torch.zeros((1, self.n_dim))
    
    @property
    def u_eq(self) -> Tensor:
        return torch.zeros((1, self.n_control))
    
    @property
    def state_limits(self) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
    
    @property
    def control_limits(self) -> tuple[Tensor, Tensor]:
        '''
        Lower and upper limits of the control inputs.
        '''
        raise NotImplementedError
    
    @property
    def n_dim(self) -> int:
        return 0
    
    @property
    def n_control(self) -> int:
        return 0
    
    @abstractmethod
    def get_mask(self, x: Tensor, type: str = 'safe') -> Tensor:
        raise NotImplementedError
    
    def sample_state_space(self, n_samples: int) -> Tensor:
        ratio = torch.zeros(n_samples, self.n_dim).uniform_(0, 1)
        lower_limit, upper_limit = self.state_limits
        return lower_limit + ratio * (upper_limit - lower_limit)
    
    def sample_with_mask(self, n_samples: int, type: str = 'safe'):
        x = self.sample_state_space(n_samples)
        while True:
            violation = ~self.get_mask(x, type)
            if violation.sum() == 0: return x
            x[violation] = self.sample_state_space(violation.sum().item())