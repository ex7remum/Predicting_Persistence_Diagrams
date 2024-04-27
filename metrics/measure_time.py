# https://pytorch.org/tutorials/recipes/recipes/benchmark.html#benchmarking-with-blocked-autorange
import torch.nn as nn
import torch
from torch.utils.benchmark import Timer

@torch.no_grad()
def calc_inference_time(model : nn.Module, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    for item in dataloader:
        src_data = item['items']
        break
        
    src_data = src_data.to(device)    
    
    t0 = Timer(stmt="model(src_data)", globals={"src_data": src_data, "model": model})
    t0.timeit(100)
    m0 = t0.blocked_autorange()

    return m0.mean * 1e6