import torch.nn as nn
from time import perf_counter
import torch

@torch.no_grad()
def calc_inference_time(model : nn.Module, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    ## warmup
    num_warmup_steps = 0
    while True:
        for item in dataloader:
            src_data = item['items']
            _ = model(src_data.to(device))
            num_steps += 1
            
            if num_warmup_steps >= 1000:
                break
                
        if num_warmup_steps >= 1000:
            break
                
    # measure time            
    start = perf_counter()
    for item in dataloader:
        src_data = item['items']
        _ = model(src_data.to(device))
        torch.cuda.synchronize()
        
    total_time = perf_counter() - start
    total_time /= len(dataloader.dataset)
    return total_time