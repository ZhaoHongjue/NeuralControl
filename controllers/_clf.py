'''

'''

import torch, numpy as np, matplotlib.pyplot as plt
from torch import Tensor
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from systems import CtrlAffSys
from ._base_controller import Controller
from . import functional as F


class CLFController(Controller):
    '''
    Class for the CLF controller
    
    Args:
    - `dynamic` (`CtrlAffSys`): control-affine system
    - `dt` (`float`): time step
    - `nominal` (`Controller`): nominal controller
    - `lamb` (`float`): lambda
    - `r_penalty` (`float`): penalty for relaxation variable
    '''
    def __init__(
        self,
        dynamic: CtrlAffSys,
        dt: float = 0.01,
        nominal: Controller = None,
        lamb: float = 1.0,
        r_penalty: float = 1.0,
        # opt_type: str = 'cvxpy',
    ):
        super().__init__(dynamic, dt)
        self.lamb, self.r_penalty = lamb, r_penalty
        self.dynamic = dynamic
        if nominal is None:
            self.nominal = ConstantController(dynamic, dt = dt)
        else:
            self.nominal = nominal
        self.qp_solver = self.init_qp_solver()
    
    def __call__(self, x: Tensor) -> Tensor:
        return self._solve_clf_qp_cvxplayers(x)[0]
    
    def v_values_and_jacob(self, x: Tensor) -> tuple[Tensor, Tensor]:
        '''
        Compute the value of the Lyapunov function and its Jacobian
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]` or `Tensor[batch_size * t_step * N_DIM]`): state vector
        
        Returns:
        - `v_values` (`Tensor[batch_size]` or `Tensor[batch_size * t_step]`): value of the Lyapunov function
        - `v_jacobs` (`Tensor[batch_size * N_DIM]` or `Tensor[batch_size * t_step * N_DIM]`): Jacobian of the Lyapunov function
        '''
        P = F.compute_sys_lyapunov_p(self.dynamic, self.nominal).to(x.device)
        if x.dim() == 2:
            v_values = torch.einsum('bi, ij, bj -> b', x, P, x).unsqueeze(-1)
            v_jacobs = torch.einsum('bi, ij -> bj', x, (P + P.T))
        elif x.dim() == 3:
            v_values = torch.einsum('bti, ij, btj -> bt', x, P, x)
            v_jacobs = torch.einsum('bti, ij -> btj', x, (P + P.T))
        return v_values, v_jacobs
    
    def v_lie_derivatives(self, x: Tensor) -> tuple[Tensor, Tensor]:
        '''
        Compute the Lie derivatives of the Lyapunov function
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]` or `Tensor[batch_size * t_step * N_DIM]`): state vector
        
        Returns:
        - `v_lie_values` (`Tensor[batch_size]` or `Tensor[batch_size * t_step]`): value of the Lie derivatives of the Lyapunov function
        - `v_lie_jacobs` (`Tensor[batch_size * N_DIM]` or `Tensor[batch_size * t_step * N_DIM]`): Jacobian of the Lie derivatives of the Lyapunov function
        '''
        f, g = self.dynamic.get_f_ang_g(x)
        _, v_jacobs = self.v_values_and_jacob(x)
        if x.dim() == 2:
            Lf_v = torch.einsum('bi, bi -> b', v_jacobs, f).unsqueeze(-1)
            Lg_v = torch.einsum('bi, bij -> bj', v_jacobs, g)
        elif x.dim() == 3:
            Lf_v = torch.einsum('bti, bti -> bt', v_jacobs, f)
            Lg_v = torch.einsum('bti, btij -> btj', v_jacobs, g)
        else: raise ValueError('Invalid input shape')
        return Lf_v, Lg_v
    
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

        constraints = [
            Lf_V_param + Lg_V_param.T @ u 
            + self.lamb * V_param - relaxation <= 0,
        ]
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
    
    def _solve_clf_qp_cvxplayers(self, x: Tensor):
        '''
        Solve the QP for the CLF controller
        
        Args:
        - `x` (`Tensor[batch_size * N_DIM]`): state vector
        
        Returns:
        - `Tensor[batch_size * N_CONTROL]`: control vector
        - `Tensor[batch_size * 1]`: relaxation variable
        '''
        v_values, _ = self.v_values_and_jacob(x)
        Lf_v, Lg_v = self.v_lie_derivatives(x)
        r_penalty = self.r_penalty * torch.ones(x.size(0), 1).to(x.device)
        u_ref_param = self.nominal(x)
        params = [v_values, Lf_v, Lg_v, r_penalty, u_ref_param]
        return self.qp_solver(*params, solver_args = {'max_iters': 1000},)