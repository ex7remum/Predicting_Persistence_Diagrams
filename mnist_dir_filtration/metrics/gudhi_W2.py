from gudhi.wasserstein import wasserstein_distance
import torch.nn as nn
import torch

@torch.no_grad()
def calc_gudhi_W2_dist(model : nn.Module, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    W2 = 0.
    for item in dataloader:
        src_data = item['items']
        tgt_pd = model(src_data.to(device))
        
        for src, tgt in zip(src_pd, tgt_pd):
            W2 += wasserstein_distance(src.cpu(), tgt.cpu(), order=1., internal_p=2.)
            
    W2 /= len(dataloader.dataset)
    return W2