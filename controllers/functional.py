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
    P = solve_continuous_lyapunov(A, Q)
    return torch.tensor(P, dtype = torch.float32)

def discrete_lin_lyapunov(A: Tensor, Q: Tensor) -> Tensor:
    '''
    Reference:
    '''
    A = A.cpu().detach().numpy()
    Q = Q.cpu().detach().numpy()
    P = solve_discrete_lyapunov(A, Q)
    return torch.tensor(P, dtype = torch.float32)

def discretize_AB(A: Tensor, B: Tensor, dt: float = 0.1, type: str = 'forward_euler') -> tuple[Tensor, Tensor]:
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

def normalize(x: Tensor, state_limits: tuple[Tensor, Tensor]) -> Tensor:
    '''
    Reference: https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/controllers/controller_utils.py#L29
    '''
    lower, upper = state_limits
    center = (lower + upper) / 2
    range = (upper - lower) / 2
    return (x - center) / range