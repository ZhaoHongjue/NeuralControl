'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/29/2025
'''

import torch, numpy as np
from torch import Tensor
from scipy.linalg import solve_continuous_are, solve_discrete_are, solve
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov


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


def continuous_lin_lyapunov(A: Tensor, Q: Tensor = None) -> Tensor:
    '''
    Reference:
    '''
    A = A.cpu().detach().numpy()
    if Q is not None: Q = Q.cpu().detach().numpy()
    else: Q = np.eye(A.shape[0])
    P = solve_continuous_lyapunov(A, -Q)
    return torch.tensor(P, dtype = torch.float32)


def discrete_lin_lyapunov(A: Tensor, Q: Tensor) -> Tensor:
    '''
    Reference:
    '''
    A = A.cpu().detach().numpy()
    Q = Q.cpu().detach().numpy()
    P = solve_discrete_lyapunov(A, -Q)
    return torch.tensor(P, dtype = torch.float32)