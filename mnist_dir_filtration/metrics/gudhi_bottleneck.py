import torch.nn as nn
from gudhi import bottleneck_distance
import torch

@torch.no_grad()
def calc_gudhi_bottleneck_dist(model : nn.Module, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    bottleneck = 0.
    for item in dataloader:
        src_data = item['items']
        tgt_pd = model(src_data.to(device))
        for src, tgt in zip(src_pd, tgt_pd):
            bottleneck += bottleneck_distance(src.cpu(), tgt.cpu())
    bottleneck /= len(dataloader.dataset)
    return bottleneck