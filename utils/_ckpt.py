'''
Author: Hongjue Zhao
Email:  hongjue2@illinois.edu
Date:   02/05/2025
'''

import torch

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    path: str
) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    
def load_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    path: str
) -> tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    checkpoint = torch.load(path, weights_only = False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']