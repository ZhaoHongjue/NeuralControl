'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/30/2025
'''

import torch
from torch import nn, Tensor

from systems import CtrlAffSys
from .. import Controller
from ._nn_certificate import NNCertificate

class NNBarrier(NNCertificate):
    @property
    def certif_type(self) -> str:
        return 'barrier'
    
    def compute_loss(self, xs: Tensor) -> Tensor:
        return 0