'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   01/29/2025
'''

import numpy as np, torch
from torch import nn

def init_seed(seed: int) -> None:
    '''
    Initialize random seed for reproducibility.
    
    Args:
    - `seed` (`int`): random seed
    
    Returns:
    - `None`
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def init_nn_weights(m: nn.Module) -> None:
    '''
    Initialize weights of a NN module using Xavier method.
    
    Args:
    - `m` (`nn.Module`): module whose weights are to be initialized
    
    Returns:
    - `None`
    '''
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
        
