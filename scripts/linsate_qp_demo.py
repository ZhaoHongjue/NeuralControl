import sys, os, time
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch, wandb
from torch.utils.data import TensorDataset, DataLoader
from lightning.fabric import Fabric

from systems import *
from controllers import *
from controllers.certificates import *
import controllers.functional as cF
from utils import save_checkpoint, init_seed

batch_size = 10000

if __name__ == '__main__':
    wandb.init(
        project = 'nn_barrier', 
        entity = 'hongjue', 
        name = 'linsate_qp_demo'
    )
    init_seed(1)
    
    ckpt_pth = './outputs/linsate_qp_demo'
    if not os.path.exists(ckpt_pth):
        os.makedirs(ckpt_pth)
    
    # Generate Data
    dynamic = LinearSatellite()
    n_samples, safe_ratio, goal_ratio, unsafe_ratio = int(1e5), 0.4, 0.3, 0.3

    print('Generating Safe Data...')
    safe_x = dynamic.sample_with_mask(int(n_samples * safe_ratio), type = 'safe')
    print('Generating Goal Data...')
    goal_x = dynamic.sample_with_mask(int(n_samples * goal_ratio), type = 'goal') 
    print('Generating Unsafe Data...')
    unsafe_x = dynamic.sample_with_mask(int(n_samples * unsafe_ratio), type = 'unsafe') 
    print('Generating Free Data...')
    free_x = dynamic.sample_state_space(int(n_samples * (1 - safe_ratio - goal_ratio - unsafe_ratio)))
    data_x = torch.cat([safe_x, goal_x, unsafe_x, free_x], dim = 0)
    print('Data Generated!')
    
    distances = data_x[:, :3].norm(dim = -1)
    print(f'0.25 <= d <= 1.5: {distances[torch.logical_and(distances <= 1.5, distances >= 0.25)].shape[0] / n_samples * 100:.2f}%')
    print(f'd > 1.5: {distances[distances > 1.5].shape[0] / n_samples * 100:.2f}%')
    print(f'd < 0.25: {distances[distances < 0.25].shape[0] / n_samples * 100:.2f}%')

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
        ), batch_size = batch_size, shuffle = True
    )

    val_iter = DataLoader(
        TensorDataset(
            val_x, 
            dynamic.get_mask(val_x, 'goal'),
            dynamic.get_mask(val_x, 'safe'),
            dynamic.get_mask(val_x, 'unsafe'),
        ), batch_size = batch_size
    )
    
    nominal_controller = LQRController(dynamic)
    nn_barrier = NNBarrier(
        dynamic, nominal_controller,
        lamb = 0.1, r_penalty = 1e4, nn_type = 'MLP',
        nn_kwargs = {'hidden_size': 256, 'n_hidden': 2, 'activation': 'ReLU',}
    )
    opt: torch.optim.Optimizer = torch.optim.Adam(
        nn_barrier.parameters(), lr = 1e-3, weight_decay = 1e-6, 
    )

    fabric = Fabric(accelerator = 'cuda', devices = [0,])
    fabric.launch()
    nn_barrier, opt = fabric.setup(nn_barrier, opt)
    train_iter, val_iter = fabric.setup_dataloaders(train_iter, val_iter)


    for epoch in range(300):
        train_loss, train_safe_loss, train_unsafe_loss, train_relaxation_loss = 0, 0, 0, 0
        for batch_idx, batch in enumerate(train_iter):
            x, goal_mask, safe_mask, unsafe_mask = batch
            start = time.time()
            
            opt.zero_grad()
            h = nn_barrier(x)
            safe_loss, unsafe_loss = cF.barrier_boundary_loss(h, safe_mask, unsafe_mask)
            safe_loss, unsafe_loss = 100 * safe_loss, 100 * unsafe_loss
            if epoch < 100: relaxation_loss = torch.tensor(0.0)
            else: relaxation_loss = cF.certif_relaxation_loss(nn_barrier, x)
            loss = safe_loss + unsafe_loss + relaxation_loss
            fabric.backward(loss)
            opt.step()
            
            finish = time.time()
            print(f'Epoch {epoch:3} | batch_idx {batch_idx:2}, Train Loss: {loss.item():.3e}, Safe Loss: {safe_loss.item():.3e}, Unsafe Loss: {unsafe_loss.item():.3e}, Relaxation Loss: {relaxation_loss.item():.3e}, Time: {finish - start:.3f}')
            train_loss += loss.item()
            train_safe_loss += safe_loss.item()
            train_unsafe_loss += unsafe_loss.item()
            train_relaxation_loss += relaxation_loss.item()        
            
        val_loss, val_safe_loss, val_unsafe_loss, val_relaxation_loss = 0, 0, 0, 0
        with torch.no_grad():
            for x, goal_mask, safe_mask, unsafe_mask in val_iter:
                
                h = nn_barrier(x)
                safe_loss, unsafe_loss = cF.barrier_boundary_loss(h, safe_mask, unsafe_mask)
                safe_loss, unsafe_loss = 100 * safe_loss, 100 * unsafe_loss
                if epoch < 100: relaxation_loss = torch.tensor(0.0)
                else: relaxation_loss = cF.certif_relaxation_loss(nn_barrier, x)
                loss = safe_loss + unsafe_loss + relaxation_loss
                
                val_loss += loss.item()
                val_safe_loss += safe_loss.item()
                val_unsafe_loss += unsafe_loss.item()
                val_relaxation_loss += relaxation_loss.item()
                
        print('-' * 150)
        print(f'Epoch {epoch:3} | Train Loss: {train_loss / len(train_iter):.3e}, Safe Loss: {train_safe_loss / len(train_iter):.3e}, Unsafe Loss: {train_unsafe_loss / len(train_iter):.3e}, Relaxation Loss: {train_relaxation_loss / len(train_iter):.3e}')
        print(f'Epoch {epoch:3} | Val Loss:   {val_loss / len(val_iter):.3e}, Safe Loss: {val_safe_loss / len(val_iter):.3e}, Unsafe Loss: {val_unsafe_loss / len(val_iter):.3e}, Relaxation Loss: {val_relaxation_loss / len(val_iter):.3e}')
        wandb.log({
            'epoch': epoch,
            'train': {
                'loss': train_loss / len(train_iter),
                'safe_loss': train_safe_loss / len(train_iter),
                'unsafe_loss': train_unsafe_loss / len(train_iter),
                'relaxation_loss': train_relaxation_loss / len(train_iter),
            },
            'val': {
                'loss': val_loss / len(val_iter),
                'safe_loss': val_safe_loss / len(val_iter),
                'unsafe_loss': val_unsafe_loss / len(val_iter),
                'relaxation_loss': val_relaxation_loss / len(val_iter),
            },
        })
        
        save_checkpoint(nn_barrier, opt, epoch, f'{ckpt_pth}/linsate_demo-epoch{epoch}.pth')
        print('=' * 150)