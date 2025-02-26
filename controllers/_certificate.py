'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/29/2025

Reference: https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/controllers/clf_controller.py
'''

from torch import Tensor

from systems import CtrlAffSys
from ._base_controller import Controller
from ._constant import ConstantController
from certificates import Certificate, QuadLyapunov


class OptCertifController(Controller):
    def __init__(
        self,
        dynamic: CtrlAffSys,
        dt: float = 0.01,
        certificate: Certificate = None,
        
    ) -> None:
        super().__init__(dynamic, dt)
        self.dynamic = dynamic

        # create nominal Lyapunov function
        if certificate is None:
            nominal_controller = ConstantController(dynamic, dt = dt)
            self.certificate = QuadLyapunov(dynamic, nominal_controller)
        else:
            self.certificate: Certificate = certificate
        
    def __call__(self, x: Tensor) -> Tensor:
        return self.certificate._solve_qp_cvxplayers(x)[0]