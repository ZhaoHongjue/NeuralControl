'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   12/29/2024
'''

from abc import ABC, abstractmethod
from torch import Tensor
from systems import CtrlAffSys


class Controller(ABC):
    '''
    Base class for controller
    '''
    def __init__(
        self, 
        dynamic: CtrlAffSys,
        dt: float = 0.1, 
        **kwargs,
    ):
        super().__init__()
        self.dynamic = dynamic
        self.dt = dt
        
    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError