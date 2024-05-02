import torch
import wandb
import losses
import numpy as np
from matplotlib import pyplot as plt
import trainer


def val_step_pd_model(model, valloader, device):
    model.eval()
    metric_chamfer = 0.
    metric_hausdorff = 0.
    val_len = len(valloader.dataset)
    log_first = True
    for batch in valloader:
        batch = trainer.move_batch_to_device(batch, device)
        X, Z = batch['items'], batch['pds']
        with torch.no_grad():
            Z_hat = model(batch)['pred_pds']
        
            metric_chamfer += losses.ChamferLoss(reduce='sum')(Z_hat, Z)
            metric_hausdorff += losses.HausdorffLoss(reduce='sum')(Z_hat, Z)

            if log_first:
                fig, axs = plt.subplots(2, 2, figsize=(20, 20))
                Z = Z.cpu().numpy()
                Z_hat = Z_hat.cpu().numpy()
                line = np.linspace(0, 1, 100)
                for i in range(4):
                    axs[i // 2, i % 2].scatter(Z[i, :, 0], Z[i, :, 1], c='b', label='real')
                    axs[i // 2, i % 2].scatter(Z_hat[i, :, 0], Z_hat[i, :, 1], c='r', label='pred')
                    axs[i // 2, i % 2].legend()
                    axs[i // 2, i % 2].plot(line, line)
                    axs[i // 2, i % 2].grid()

                wandb.log({"PD progress": plt})
                log_first = False
    wandb.log({'val_chamfer': metric_chamfer / val_len, 'val_hausdorff': metric_hausdorff / val_len})
