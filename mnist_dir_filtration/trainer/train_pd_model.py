import torch
import wandb
from torch.utils.data import DataLoader
import losses

def val_step(model, valloader, device):
    model.eval()
    metric_chamfer = 0.
    metric_hausdorff = 0.
    val_len = len(valloader.dataset)
    for item in valloader:
        X, Z, v = item['items'], item['pds'], item['labels']
        with torch.no_grad():
            Z = Z[..., :2].to(torch.float32).to(device)
            Z_hat = model(X.to(device))
        
        metric_chamfer += losses.ChamferLoss(reduce='sum')(Z_hat, Z)
        metric_hausdorff += losses.HausdorffLoss(reduce='sum')(Z_hat, Z)
    wandb.log({'val_chamfer': metric_chamfer / val_len, 'val_hausdorff': metric_hausdorff / val_len})

def train_loop_pd(model, trainloader, valloader, optimizer, loss_fn, device, scheduler=None, n_epochs=25, clip_norm=None, seed=0):
    torch.manual_seed(seed)
    
    for _ in range(n_epochs):
        model.train()
        for item in trainloader:
            X, Z, v = item['items'], item['pds'], item['labels']
            optimizer.zero_grad()
            
            Z = Z[..., :2].to(torch.float32).to(device)
            Z_hat = model(X.to(device))
            loss = loss_fn(Z_hat, Z)
            
            loss.backward()
            
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()

            total_norm = 0
            parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            lr = optimizer.param_groups[0]['lr']
            wandb.log({'loss_pd': loss, 'grad_norm_pd': total_norm, 'learning_rate_pd': lr})
        
        val_step(model, valloader, device)
    return model
