from gudhi.wasserstein import wasserstein_distance
import torch.nn as nn
import torch
from trainer import move_batch_to_device


@torch.no_grad()
def calc_gudhi_W2_dist(model: nn.Module, dataloader):
    device = next(model.parameters()).device
    model.eval()
    W2 = 0.
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        src_pd = batch['pds'].to(torch.float32)
        tgt_pd = model(batch)['pred_pds']
        
        for src, tgt in zip(src_pd, tgt_pd):
            W2 += wasserstein_distance(src.cpu(), tgt.cpu(), order=1., internal_p=2.)
            
    W2 /= len(dataloader.dataset)
    return W2
