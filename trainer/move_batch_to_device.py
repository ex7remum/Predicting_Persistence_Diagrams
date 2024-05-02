import torch


def move_batch_to_device(batch, device: torch.device):
    for tensor_for_gpu in ['labels', 'pds', 'items', 'lengths', 'mask', 'pis']:
        if tensor_for_gpu in batch and batch[tensor_for_gpu] is not None:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
    return batch