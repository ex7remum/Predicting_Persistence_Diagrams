import torch
import torch.nn as nn
from torch.nn import MSELoss
from trainer import move_batch_to_device


@torch.no_grad()
def calc_pie_from_pi(model: nn.Module, dataloader, pimgr=None):
    device = next(model.parameters()).device
    model.eval()
    
    total_pie = 0.
    mse = MSELoss(reduction='sum')
    
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        PI_pred = model(batch)['pred_pis']
        PI_real = batch['pis']

        total_pie += mse(PI_pred, PI_real)
       
    return total_pie / len(dataloader.dataset)


@torch.no_grad()
def calc_pie_from_pd(model: nn.Module, dataloader, pimgr):
    device = next(model.parameters()).device
    model.eval()
    
    total_pie = 0.
    mse = MSELoss(reduction='sum')
    
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)

        PI_real = batch['pis']
        PD_pred = model(batch)['pred_pds']
        
        PI_pred = torch.from_numpy(pimgr.fit_transform(PD_pred.cpu())).to(torch.float32).to(device)
        PI_pred = PI_pred / torch.max(PI_pred, dim=1, keepdim=True)[0]

        total_pie += mse(PI_pred, PI_real)
       
    return total_pie / len(dataloader.dataset)
