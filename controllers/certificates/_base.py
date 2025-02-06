'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/30/2025

Reference: https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/controllers/clf_controller.py
'''

from abc import ABC, abstractmethod

import torch
from torch import Tensor

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from systems import CtrlAffSys
from .. import Controller, ConstantController


class Certificate(ABC):
    def __init__(
        self,
        dynamic: CtrlAffSys,
        nominal_controller: Controller = None,
        lamb: float = 1.0,
        r_penalty: float = 1.0,
        **kwargs,
    ) -> None:
        self.dynamic = dynamic
        if nominal_controller is None:
            self.nominal_controller = ConstantController(dynamic, dt = dynamic.dt)
        else:
            self.nominal_controller = nominal_controller
        self.lamb, self.r_penalty = lamb, r_penalty
        self.qp_solver = self.init_qp_solver()
    
    def __call__(self, xs: Tensor) -> Tensor:
        return torch.vmap(self._value)(xs).unsqueeze(-1)
    
    @property
    def certif_type(self) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def _value(self, x: Tensor) -> Tensor:
        '''
        Compute the value of the Lyapunov function
        
        Args:
        - `x` (`Tensor[N_DIM]`): state vector
        
        Returns:
        - `v` (`Tensor[1]`): value of the Lyapunov function
        '''
        raise NotImplementedError
    
    
    def compute_jacobian(self, xs: Tensor) -> Tensor:
        '''
        Compute the Jacobian of the Lyapunov function
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): state vector
        '''
        return torch.vmap(torch.func.grad(self._value))(xs)
        
        
    def compute_lie_deriv(self, x: Tensor) -> Tensor:
        '''
        Compute the Lie derivative of the certificate function
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): state vector
        '''
        f, g = self.dynamic.get_f_ang_g(x)
        v_jacob = self.compute_jacobian(x)
        Lf_v = torch.einsum('bi, bi -> b', f, v_jacob).unsqueeze(-1)
        Lg_v = torch.einsum('bij, bi -> bj', g, v_jacob)
        return Lf_v, Lg_v
    
    def init_qp_solver(self):
        '''
        Initialize the QP solver
        
        Returns:
        - `CvxpyLayer`: QP solver'''
        u = cp.Variable(self.dynamic.n_control)
        relaxation = cp.Variable(1, nonneg = True)
        V_param = cp.Parameter(1, nonneg = self.certif_type == 'lyapunov')
        Lf_V_param = cp.Parameter(1)
        Lg_V_param = cp.Parameter(self.dynamic.n_control)
        r_penalty_param = cp.Parameter(1, nonneg = True)
        u_ref_param = cp.Parameter(self.dynamic.n_control)
        certificate_cond = Lf_V_param + Lg_V_param @ u + self.lamb * V_param
        
        if self.certif_type == 'lyapunov':
            constraints = [certificate_cond <= relaxation]
        elif self.certif_type == 'barrier':
            constraints = [certificate_cond >= -relaxation]
        else:
            raise ValueError('Unknown certificate type')
            
        lower_limits, upper_limits = self.dynamic.control_limits
        for i in range(self.dynamic.n_control):
            constraints.append(u[i] >= lower_limits[i])
            constraints.append(u[i] <= upper_limits[i])

        obj_expr = cp.norm(u - u_ref_param)**2 + cp.multiply(r_penalty_param, relaxation)
        obj = cp.Minimize(obj_expr)

        problem = cp.Problem(obj, constraints)
        assert problem.is_dpp(), 'Problem is not DPP'
        varaibles = [u, relaxation]
        parameters = [V_param, Lf_V_param, Lg_V_param, r_penalty_param, u_ref_param]
        return CvxpyLayer(problem, parameters, varaibles)
    
    def _solve_qp_cvxplayers(self, xs: Tensor) -> Tensor:
        '''
        Solve the QP for the CLF controller
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): state vector
        
        Returns:
        - `Tensor[batch_size * N_CONTROL]`: control vector
        - `Tensor[batch_size * 1]`: relaxation variable
        '''
        v_values = self(xs)
        Lf_v, Lg_v = self.compute_lie_deriv(xs)
        r_penalty = self.r_penalty * torch.ones(xs.size(0), 1).to(xs.device)
        us_ref = self.nominal_controller(xs)
        params = [v_values, Lf_v, Lg_v, r_penalty, us_ref]
        return self.qp_solver(*params, solver_args = {'solve_method': 'Clarabel'},) # 
    
    def get_relaxation(self, x: Tensor) -> Tensor:
        return self._solve_qp_cvxplayers(x)[1]