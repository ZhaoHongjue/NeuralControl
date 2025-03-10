'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/29/2025
'''

import torch, numpy as np
from torch import Tensor
from scipy.linalg import solve_continuous_are, solve_discrete_are, solve
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov

from systems import CtrlAffSys
from controllers import Controller


def continuous_lqr(A: Tensor, B: Tensor, Q: Tensor, R: Tensor) -> Tensor:
    '''
    Reference:
    '''
    A = A.cpu().detach().numpy()
    B = B.cpu().detach().numpy()
    Q = Q.cpu().detach().numpy()
    R = R.cpu().detach().numpy()
    P = solve_continuous_are(A, B, Q, R)
    return torch.tensor(solve(R, B.T @ P), dtype = torch.float32)


def discrete_lqr(A: Tensor, B: Tensor, Q: Tensor, R: Tensor) -> Tensor:
    '''
    Reference:
    '''
    A = A.cpu().detach().numpy()
    B = B.cpu().detach().numpy()
    Q = Q.cpu().detach().numpy()
    R = R.cpu().detach().numpy()
    P = solve_discrete_are(A, B, Q, R)
    return torch.tensor(solve(R + B.T @ P @ B, B.T @ P @ A), dtype = torch.float32)


def continuous_lin_lyapunov_p(A: Tensor, Q: Tensor = None) -> Tensor:
    '''
    Reference:
    '''
    A = A.cpu().detach().numpy()
    if Q is not None: Q = Q.cpu().detach().numpy()
    else: Q = np.eye(A.shape[0])
    P = solve_continuous_lyapunov(A, -Q)
    return torch.tensor(P, dtype = torch.float32)


def discrete_lin_lyapunov_p(A: Tensor, Q: Tensor) -> Tensor:
    '''
    Reference:
    '''
    A = A.cpu().detach().numpy()
    Q = Q.cpu().detach().numpy()
    P = solve_discrete_lyapunov(A, -Q)
    return torch.tensor(P, dtype = torch.float32)


def compute_sys_lyapunov_p(
    dynamic: CtrlAffSys,
    controller: Controller,
    sys_type: str = 'continuous'
):
    if type(controller).__name__ == 'LQRController':
        A_closed = dynamic.A - dynamic.B @ controller.K
        Q = controller.Q
    else:
        A_closed = dynamic.A
        Q = torch.eye(dynamic.n_dim)
    
    if sys_type == 'continuous':
        return continuous_lin_lyapunov_p(A_closed, Q)
    elif sys_type == 'discrete':
        A_d, _ = discretize_AB(dynamic.A, dynamic.B, dynamic.dt)
        return discrete_lin_lyapunov_p(A_d, Q)
    else:
        raise ValueError('Type must be either continuous or discrete')
    
    
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
    state_limits: tuple[Tensor, Tensor],
    scale: float = 1.0
) -> Tensor:
    '''
    Reference: https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/controllers/controller_utils.py#L29
    '''
    lower, upper = state_limits
    center = (lower + upper) / 2
    range = (upper - lower) / 2
    range /= scale
    return (x - center.to(x.device)) / range.to(x.device)