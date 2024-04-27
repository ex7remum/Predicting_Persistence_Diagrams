import torch
import wandb
from torch.nn import MSELoss, BCELoss
import torchvision

def val_step(model, valloader, pimgr, device):
    model.eval()
    pie = 0.
    mse = MSELoss(reduction='sum')
    val_len = len(valloader.dataset)
    
    log_img = True
    
    for item in valloader:
        X, Z, v = item['items'], item['pds'], item['labels']
        with torch.no_grad():
            Z = Z[..., :2].to(torch.float32)
            PI_pred = model(X.to(device))
            PI_real = torch.from_numpy(pimgr.fit_transform(Z)).to(torch.float32).to(device)
            PI_real = PI_real / PI_real.max(dim=1, keepdim=True)[0]
            pie += mse(PI_pred, PI_real)
            
        if log_img:
            PI_real = PI_real.view(-1, 1, 50, 50)
            PI_pred = PI_pred.view(-1, 1, 50, 50)
            
            PI_real_grid = torchvision.utils.make_grid(PI_real)
            PI_pred_grid = torchvision.utils.make_grid(PI_pred)
            
            images_real = wandb.Image(PI_real_grid, caption="Real PI")
            images_pred = wandb.Image(PI_pred_grid, caption="Pred PI")
            
            wandb.log({"Real PI": images_real})
            wandb.log({"Pred PI": images_pred})
            
            # log only first batch
            log_img = False

    wandb.log({'val_pie': pie / val_len})
    

def train_loop_pi(model, trainloader, valloader, optimizer, pimgr, device, scheduler=None, n_epochs=25, clip_norm=None, seed=0):
    torch.manual_seed(seed)
    loss_fn = BCELoss()
    
    for _ in range(n_epochs):
        model.train()
        for item in trainloader:
            X, Z, v = item['items'], item['pds'], item['labels']
            optimizer.zero_grad()
            
            Z = Z[..., :2].to(torch.float32)
            PI_pred = model(X.to(device))
            PI_real = torch.from_numpy(pimgr.fit_transform(Z)).to(torch.float32).to(device)
            PI_real = PI_real / PI_real.max(dim=1, keepdim=True)[0]

            loss = loss_fn(PI_pred, PI_real)
            
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
            wandb.log({'loss_pi': loss, 'grad_norm_pi': total_norm, 'learning_rate_pi': lr})
        
        val_step(model, valloader, pimgr, device)
    return model