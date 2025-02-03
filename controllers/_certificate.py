'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/29/2025

Reference: https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/controllers/clf_controller.py
'''

import torch, numpy as np, matplotlib.pyplot as plt
from torch import Tensor
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from systems import CtrlAffSys
from ._base_controller import Controller
from ._constant import ConstantController
from . import functional as F
from .certificates import Certificate, QuadLyapunov


class CertifController(Controller):
    def __init__(
        self,
        dynamic: CtrlAffSys,
        dt: float = 0.01,
        nominal_controller: Controller = None,
        certificate: Certificate = None,
        lamb: float = 1.0,
        r_penalty: float = 1.0,
    ) -> None:
        super().__init__(dynamic, dt)
        self.lamb, self.r_penalty = lamb, r_penalty
        self.dynamic = dynamic
        
        # create nominal controller
        if nominal_controller is None:
            self.nominal_controller = ConstantController(dynamic, dt = dt)
        else:
            self.nominal_controller = nominal_controller
            
        # create nominal Lyapunov function
        if certificate is None:
            self.certificate = QuadLyapunov(dynamic, self.nominal_controller)
        else:
            self.certificate: Certificate = certificate
            
        self.qp_solver = self.init_qp_solver()
        
    def __call__(self, x: Tensor) -> Tensor:
        return self._solve_qp_cvxplayers(x)[0]
    
    def get_relaxation(self, x: Tensor) -> Tensor:
        return self._solve_qp_cvxplayers(x)[1]
        
    def init_qp_solver(self):
        '''
        Initialize the QP solver
        
        Returns:
        - `CvxpyLayer`: QP solver'''
        u = cp.Variable(self.dynamic.n_control)
        relaxation = cp.Variable(1, nonneg = True)
        V_param = cp.Parameter(1, nonneg = True)
        Lf_V_param = cp.Parameter(1)
        Lg_V_param = cp.Parameter(self.dynamic.n_control)
        r_penalty_param = cp.Parameter(1, nonneg = True)
        u_ref_param = cp.Parameter(self.dynamic.n_control)
        constraint_expr = Lf_V_param + Lg_V_param.T @ u + self.lamb * V_param - relaxation
        
        if self.certificate.certif_type == 'lyapunov':
            constraints = [constraint_expr <= 0]
        elif self.certificate.certif_type == 'barrier':
            constraints = [constraint_expr >= 0]
        else:
            raise ValueError('Unknown certificate type')
            
        # lower_limits, upper_limits = self.dynamic.control_limits
        # for i in range(self.dynamic.n_control):
        #     constraints.append(u[i] >= lower_limits[i])
        #     constraints.append(u[i] <= upper_limits[i])

        obj_expr = cp.sum_squares(u - u_ref_param) + cp.multiply(r_penalty_param, relaxation)
        obj = cp.Minimize(obj_expr)

        problem = cp.Problem(obj, constraints)
        assert problem.is_dpp()
        varaibles = [u, relaxation]
        parameters = [V_param, Lf_V_param, Lg_V_param, r_penalty_param, u_ref_param]
        return CvxpyLayer(problem, parameters, varaibles)
    
    def _solve_qp_cvxplayers(self, x: Tensor):
        '''
        Solve the QP for the CLF controller
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): state vector
        
        Returns:
        - `Tensor[batch_size * N_CONTROL]`: control vector
        - `Tensor[batch_size * 1]`: relaxation variable
        '''
        v_values = self.certificate(x)
        Lf_v, Lg_v = self.certificate.compute_lie_deriv(x)
        r_penalty = self.r_penalty * torch.ones(x.size(0), 1).to(x.device)
        u_ref_param = self.nominal_controller(x)
        params = [v_values, Lf_v, Lg_v, r_penalty, u_ref_param]
        return self.qp_solver(*params, solver_args = {'max_iters': 1000},)