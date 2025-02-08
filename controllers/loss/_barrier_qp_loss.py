'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/05/2025
'''

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from ..certificates import Certificate, NNCertificate, NNBarrier
from ..functional import barrier_boundary_loss, barrier_relaxation_loss


class Barrier_QP_Loss:
    def __init__(
        self, 
        nn_barrier: NNBarrier,
        shape_epoch: int = 100,
        lambs: list[float] = [100., 100., 1.0],
        reduction: str = 'mean',
        **kwargs
    ):
        self.nn_barrier = nn_barrier
        self.shape_epoch = shape_epoch
        self.lambs = lambs
        self.reduction = reduction
    
    def __call__(
        self, 
        xs: Tensor,
        goal_mask: Tensor,
        safe_mask: Tensor,
        unsafe_mask: Tensor,
        current_epoch: int,
        **kwargs
    ) -> Tensor:
        hs: Tensor = self.nn_barrier(xs)
        safe_loss, unsafe_loss = barrier_boundary_loss(
            hs, safe_mask, unsafe_mask, reduction = self.reduction
        )
        if current_epoch > self.shape_epoch:
            relaxation_loss = barrier_relaxation_loss(
                self.nn_barrier, xs, reduction = self.reduction
            )
        else: relaxation_loss = 0.0
        return self.lambs[0] * safe_loss, self.lambs[1] * unsafe_loss, self.lambs[2] * relaxation_loss
        
        
        




