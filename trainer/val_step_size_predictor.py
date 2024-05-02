import torch
import wandb
from torch.nn import MSELoss
import trainer


def val_step_size_predictor(model, valloader, device):
    model.eval()
    loss_fn = MSELoss(reduction='sum')
    val_len = len(valloader.dataset)
    total_mse = 0

    for batch in valloader:
        batch = trainer.move_batch_to_device(batch, device)
        lengths = batch['lengths']
        with torch.no_grad():
            pred_size = model(batch)['pred_size'].squeeze(-1)  # (batch_size, )
            total_mse += loss_fn(pred_size, lengths)
    total_mse /= val_len
    wandb.log({'val_mse_size': total_mse})
