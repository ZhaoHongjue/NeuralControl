'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/29/2025
'''

import torch, numpy as np, torchode as tode
from torch import Tensor
from scipy.linalg import solve_continuous_are, solve_discrete_are, solve
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov

from systems import CtrlAffSys
from .._base_controller import Controller


def simulate_con_system(
    dynamic: CtrlAffSys,
    controller: Controller,
    batch_x0: Tensor,
    T: float = 10.,
    dt: float = 0.01,
) -> tuple[Tensor, Tensor]:
    '''
    Simulate a control-affine system with a controller.
    
    Args:
    - `dynamic` (`CtrlAffSys`): control-affine system
    - `controller` (`Controller`): controller
    - `x` (`Tensor[batch_size * N_DIM]`): State vectors. 
    - `T` (`float`): Simulation time.
    - `dt` (`float`): Time step.
    '''
    ode_func = lambda t, x: dynamic.closed_loop_dynamics(x, controller(x))
    term = tode.ODETerm(ode_func)
    step_method = tode.Tsit5(term)
    step_size_controller = tode.IntegralController(atol=1e-6, rtol=1e-3, term=term)
    solver = tode.AutoDiffAdjoint(step_method, step_size_controller)
    ts_eval = torch.broadcast_to(torch.arange(0, T + dt, dt), (batch_x0.shape[0], int(T/dt) + 1))
    problem = tode.InitialValueProblem(y0 = batch_x0, t_eval = ts_eval)
    sol = solver.solve(problem)
    return sol.ts, sol.ys