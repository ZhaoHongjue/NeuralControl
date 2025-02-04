import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
from torch.utils.data import TensorDataset, DataLoader
from lightning.fabric import Fabric

from systems import *
from controllers import *
from controllers.certificates import *
import utils

utils.init_seed(0)

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    
def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']

if __name__ == '__main__':
    utils.init_seed(1)
    
    ckpt_pth = './outputs'
    if not os.path.exists(ckpt_pth):
        os.makedirs(ckpt_pth)
    
    # Generate Data
    dynamic = LinearSatellite()
    n_samples, goal_ratio, safe_ratio = int(1e5), 0.4, 0.2

    print('Generating Goal Data...')
    goal_x = dynamic.sample_with_mask(int(n_samples * goal_ratio), type = 'goal') 
    print('Generating Safe Data...')
    safe_x = dynamic.sample_with_mask(int(n_samples * safe_ratio), type = 'safe')
    print('Generating Free Data...')
    free_x = dynamic.sample_state_space(n_samples - goal_x.shape[0] - safe_x.shape[0])
    data_x = torch.cat([goal_x, safe_x, free_x], dim = 0)

    random_indices = torch.randperm(data_x.shape[0])
    split_idx = int(0.8 * data_x.shape[0])
    train_x = data_x[random_indices[:split_idx]]
    val_x = data_x[random_indices[split_idx:]]

    train_iter = DataLoader(
        TensorDataset(
            train_x, 
            dynamic.get_mask(train_x, 'goal'),
            dynamic.get_mask(train_x, 'safe'),
            dynamic.get_mask(train_x, 'unsafe'),
        ), batch_size = 5120, shuffle = True
    )

    val_iter = DataLoader(
        TensorDataset(
            val_x, 
            dynamic.get_mask(val_x, 'goal'),
            dynamic.get_mask(val_x, 'safe'),
            dynamic.get_mask(val_x, 'unsafe'),
        ), batch_size = 5120
    )
    
    nominal_controller = LQRController(dynamic)
    nn_barrier = NNBarrier(
        dynamic, nominal_controller,
        lamb = 0.1, r_penalty = 1e4, nn_type = 'MLP',
        nn_kwargs = {'hidden_size': 256, 'n_hidden': 2}
    )
    opt: torch.optim.Optimizer = torch.optim.Adam(nn_barrier.parameters(), lr = 1e-4)

    fabric = Fabric(accelerator = 'cuda', devices = [0,])
    fabric.launch()
    model, opt = fabric.setup(nn_barrier, opt)
    train_iter, val_iter = fabric.setup_dataloaders(train_iter, val_iter)


    for epoch in range(200):
        for batch_idx, batch in enumerate(train_iter):
            x, goal_mask, safe_mask, unsafe_mask = batch
            opt.zero_grad()
            safe_violation_loss, unsafe_violation_loss = nn_barrier.compute_violation_loss(x, goal_mask, safe_mask, unsafe_mask)
            if epoch < 100:
                relaxation_loss = torch.tensor(0.0)
                loss = safe_violation_loss + unsafe_violation_loss
            else:
                relaxation_loss = nn_barrier.compute_relaxation_loss(x)
                loss = safe_violation_loss + unsafe_violation_loss + relaxation_loss
            fabric.backward(loss)
            opt.step()
            print(f'Epoch {epoch}, batch_idx {batch_idx} | Train Loss: {loss.item():.3e}, Safe Loss: {safe_violation_loss.item():.3e}, Unsafe Loss: {unsafe_violation_loss.item():.3e}, Relaxation Loss: {relaxation_loss.item():.3e}')

        val_loss = 0
        with torch.no_grad():
            for x, goal_mask, safe_mask, unsafe_mask in val_iter:
                safe_violation_loss, unsafe_violation_loss = nn_barrier.compute_violation_loss(x, goal_mask, safe_mask, unsafe_mask)
                if epoch < 100:
                    relaxation_loss = torch.tensor(0.0)
                    loss = safe_violation_loss + unsafe_violation_loss
                else:
                    relaxation_loss = nn_barrier.compute_relaxation_loss(x)
                    loss = safe_violation_loss + unsafe_violation_loss + relaxation_loss
                val_loss += loss.item()
        print(f'Epoch {epoch} | Val Loss: {val_loss / len(val_iter)}')
        
        save_checkpoint(nn_barrier, opt, epoch, f'{ckpt_pth}/linsate_demo-epoch{epoch}.pth')
        print('-' * 100)