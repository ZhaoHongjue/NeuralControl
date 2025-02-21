import sys, os, time, argparse
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch, wandb
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from lightning.fabric import Fabric

from systems import *
from controllers import *
from controllers.certificates import *
import controllers.functional as cF
from utils import save_checkpoint, init_seed

batch_size = 1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lamb', type = float, default = 0.5)
    parser.add_argument('--cuda', type = int, default = 0)
    args = parser.parse_args()
    
    wandb.init(
        project = 'nn_lyapunov', 
        entity = 'hongjue', 
        name = f'inverted_pendulum_qp_demo-lamb{args.lamb}',
        config = {
            'lamb': args.lamb,
        }
    )
    init_seed(1)

    ckpt_pth = f'./outputs/inverted_pendulum_qp_demo-lamb{args.lamb}'
    if not os.path.exists(ckpt_pth):
        os.makedirs(ckpt_pth)
        
    # Generate Data
    dynamic = InvertedPendulum()
    data_x = dynamic.sample_state_space(int(1e4))
    goal_point = dynamic.goal_point

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
    
    nominal_controller = ConstantController(dynamic)
    nn_lyap = NNLyapunov(
        dynamic, nominal_controller,
        lamb = args.lamb, r_penalty = 1e5, nn_type = 'QuadNN',
        nn_kwargs = {'mlp_output_size': 32, 'hidden_size': 256, 'layer_num': 2, 'activation': 'ReLU',}
    )
    opt: torch.optim.Optimizer = torch.optim.Adam(
        nn_lyap.parameters(), lr = 1e-4, weight_decay = 1e-6, 
    )

    fabric = Fabric(accelerator = 'cuda', devices = [args.cuda,])
    fabric.launch()
    nn_lyap, opt = fabric.setup(nn_lyap, opt)
    train_iter, val_iter = fabric.setup_dataloaders(train_iter, val_iter)
    
    for epoch in range(300):
        train_loss, train_goal_loss, train_deriv_loss, train_relaxation_loss = 0, 0, 0, 0
        for batch_idx, batch in enumerate(train_iter):
            x, goal_mask, safe_mask, unsafe_mask = batch
            goal_point = goal_point.to(x.device)
            
            start = time.time()
            opt.zero_grad()
            goal_loss: Tensor = 10 * nn_lyap(goal_point).mean()
            deriv_loss, relaxation_loss = cF.lyap_qp_loss(nn_lyap, x)
            deriv_loss = 1000 * deriv_loss
            relaxation_loss = 1000 * relaxation_loss
            loss = goal_loss + deriv_loss + relaxation_loss
            fabric.backward(loss)
            opt.step()
            finish = time.time()
            
            print(f'Epoch {epoch:3} | batch_idx {batch_idx:2}, Train Loss: {loss.item():.3e}, Goal Loss: {goal_loss.item():.3e}, Deriv Loss: {deriv_loss.item():.3e}, Relaxation Loss: {relaxation_loss.item():.3e}, Time: {finish - start:.3f}')
            train_loss += loss.item()
            train_goal_loss += goal_loss.item()
            train_deriv_loss += deriv_loss.item()
            train_relaxation_loss += relaxation_loss.item()
            
        with torch.no_grad():
            val_loss, val_goal_loss, val_deriv_loss, val_relaxation_loss = 0, 0, 0, 0
            for x, goal_mask, safe_mask, unsafe_mask in val_iter:
                goal_point = goal_point.to(x.device)
                goal_loss: Tensor = 10 * nn_lyap(goal_point).mean()
                deriv_loss, relaxation_loss = cF.lyap_qp_loss(nn_lyap, x)
                deriv_loss = 1000 * deriv_loss
                relaxation_loss = 1000 * relaxation_loss
                loss = goal_loss + deriv_loss + relaxation_loss
                
                val_loss += loss.item()
                val_goal_loss += goal_loss.item()
                val_deriv_loss += deriv_loss.item()
                val_relaxation_loss += relaxation_loss.item()
            
        print('-' * 150)
        print(f'Epoch {epoch:3} | Train Loss: {train_loss / len(train_iter):.3e}, Goal Loss: {train_goal_loss / len(train_iter):.3e}, Deriv Loss: {train_deriv_loss / len(train_iter):.3e}, Relaxation Loss: {train_relaxation_loss / len(train_iter):.3e}')
        print(f'Epoch {epoch:3} | Val Loss:   {val_loss / len(val_iter):.3e}, Goal Loss: {val_goal_loss / len(val_iter):.3e}, Deriv Loss: {val_deriv_loss / len(val_iter):.3e}, Relaxation Loss: {val_relaxation_loss / len(val_iter):.3e}')
        wandb.log({
            'epoch': epoch,
            'train': {
                'loss': train_loss / len(train_iter),
                'goal_loss': train_goal_loss / len(train_iter),
                'deriv_loss': train_deriv_loss / len(train_iter),
                'relaxation_loss': train_relaxation_loss / len(train_iter),
            },
            'val': {
                'loss': val_loss / len(val_iter),
                'goal_loss': val_goal_loss / len(val_iter),
                'deriv_loss': val_deriv_loss / len(val_iter),
                'relaxation_loss': val_relaxation_loss / len(val_iter),
            },
        })  
            
        save_checkpoint(nn_lyap, opt, epoch, f'{ckpt_pth}/inverted_pendulum_qp_demo-epoch{epoch}.pt')
        print('=' * 150)
    