import torch.nn as nn
from gudhi import bottleneck_distance
import torch
from trainer import move_batch_to_device


@torch.no_grad()
def calc_gudhi_bottleneck_dist(model: nn.Module, dataloader):
    device = next(model.parameters()).device
    model.eval()
    bottleneck = 0.
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        src_pd = batch['pds'].to(torch.float32)
        tgt_pd = model(batch)['pred_pds']

        for src, tgt in zip(src_pd, tgt_pd):
            bottleneck += bottleneck_distance(src.cpu(), tgt.cpu())
    bottleneck /= len(dataloader.dataset)
    return bottleneck
