import torch
import wandb
from torch.nn import CrossEntropyLoss
import torchvision


def val_step(model, valloader, real_pd, device, model_pd=None):
    model.eval()
    correct = 0.
    val_len = len(valloader.dataset)
    
    for item in valloader:
        X, Z, v = item['items'], item['pds'], item['labels']
        with torch.no_grad():
            Z = Z[..., :2].to(torch.float32)
            if real_pd:
                pred_pds = Z.detach().clone().to(device)
            else:
                pred_pds = model_pd(X.to(device))

            logits = model(pred_pds)
            correct += (v.to(device) == torch.argmax(logits, axis=1)).sum()
            
    val_pd_acc = correct / val_len
    if real_pd:
        wandb.log({'val_acc_real': val_pd_acc})
    else:
        wandb.log({'val_acc_pred': val_pd_acc})
    

def train_loop_class(model, trainloader, valloader, optimizer, real_pd, device, model_pd=None, scheduler=None, 
               n_epochs=25, clip_norm=None, seed=0):
    torch.manual_seed(seed)
    loss_fn = CrossEntropyLoss()
    
    for _ in range(n_epochs):
        model.train()
        for item in trainloader:
            X, Z, v = item['items'], item['pds'], item['labels']
            Z = Z[..., :2].to(torch.float32)
            
            optimizer.zero_grad()
           
            with torch.no_grad():
                if real_pd:
                    pred_pds = Z.detach().clone().to(device)
                else:
                    pred_pds = model_pd(X.to(device))
            
            logits = model(pred_pds)            
            loss = loss_fn(logits, v.to(device))
            correct = (v.to(device) == torch.argmax(logits, axis=1)).sum()
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
            
            if real_pd:
                name = 'train_acc_real'
            else:
                name = 'train_acc_pred'
            
            wandb.log({'loss_class': loss, 'grad_norm_class': total_norm, 'learning_rate_class': lr, name: correct / len(Z)})
        
        val_step(model, valloader, real_pd, device, model_pd)
    return model