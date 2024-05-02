# https://pytorch.org/tutorials/recipes/recipes/benchmark.html#benchmarking-with-blocked-autorange
import torch.nn as nn
import torch
from torch.utils.benchmark import Timer
from trainer import move_batch_to_device


@torch.no_grad()
def calc_inference_time(model: nn.Module, dataloader):
    device = next(model.parameters()).device
    model.eval()
    
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        break
    
    t0 = Timer(stmt="model(batch)", globals={"batch": batch, "model": model})
    t0.timeit(100)
    m0 = t0.blocked_autorange()

    return m0.mean * 1e6
