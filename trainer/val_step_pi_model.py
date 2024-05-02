import torch
import wandb
from torch.nn import MSELoss
import torchvision
import trainer


def val_step_pi_model(model, valloader, device):
    model.eval()
    pie = 0.
    mse = MSELoss(reduction='sum')
    val_len = len(valloader.dataset)
    
    log_img = True
    
    for batch in valloader:
        batch = trainer.move_batch_to_device(batch, device)

        X, Z, PI_real = batch['items'], batch['pds'], batch['pis']
        with torch.no_grad():
            PI_pred = model(batch)['pred_pis']
            pie += mse(PI_pred, PI_real)
            
        if log_img:
            PI_real = PI_real.view(-1, 1, 50, 50)
            PI_pred = PI_pred.view(-1, 1, 50, 50)
            
            PI_real_grid = torchvision.utils.make_grid(PI_real)
            PI_pred_grid = torchvision.utils.make_grid(PI_pred)
            
            images_real = wandb.Image(PI_real_grid, caption="Real PI")
            images_pred = wandb.Image(PI_pred_grid, caption="Pred PI")
            
            wandb.log({"Real PI": images_real})
            wandb.log({"Pred PI": images_pred})
            
            # log only first batch
            log_img = False

    wandb.log({'val_pie': pie / val_len})
