import torch
import torch.nn as nn
from torch.nn import MSELoss

@torch.no_grad()
def calc_pie_from_pi(model : nn.Module, dataloader, pimgr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    total_pie = 0.
    mse = MSELoss(reduction = 'sum')
    
    for item in dataloader:
        X, Z, v = item['items'], item['pds'], item['labels']
        Z = Z[..., :2].to(torch.float32)
        PI_pred = model(X.to(device))
        PI_real = torch.from_numpy(pimgr.fit_transform(Z)).to(torch.float32).to(device)
        
        PI_real = PI_real / torch.max(PI_real, dim=1, keepdim=True)[0]
        total_pie += mse(PI_pred, PI_real)
       
    return total_pie / len(dataloader.dataset)



@torch.no_grad()
def calc_pie_from_pd(model : nn.Module, dataloader, pimgr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    total_pie = 0.
    mse = MSELoss(reduction = 'sum')
    
    for item in dataloader:
        X, Z, v = item['items'], item['pds'], item['labels']
        Z = Z[..., :2].to(torch.float32)
        PD_pred = model(X.to(device))
        
        PI_pred = torch.from_numpy(pimgr.fit_transform(PD_pred.cpu())).to(torch.float32).to(device)
        PI_real = torch.from_numpy(pimgr.fit_transform(Z)).to(torch.float32).to(device)
        
        PI_pred = PI_pred / torch.max(PI_pred, dim=1, keepdim=True)[0]
        PI_real = PI_real / torch.max(PI_real, dim=1, keepdim=True)[0]
        total_pie += mse(PI_pred, PI_real)
       
    return total_pie / len(dataloader.dataset)