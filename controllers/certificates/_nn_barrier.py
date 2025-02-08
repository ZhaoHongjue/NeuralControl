'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/30/2025
'''

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from ._nn_certificate import NNCertificate

class NNBarrier(NNCertificate):
    @property
    def certif_type(self) -> str:
        return 'barrier'