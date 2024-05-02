import torch
import wandb
import trainer


def val_step_classificator(model, valloader, device):
    model.eval()
    correct = 0.
    val_len = len(valloader.dataset)
    
    for batch in valloader:
        batch = trainer.move_batch_to_device(batch, device)
        labels = batch['labels']
        with torch.no_grad():
            logits = model(batch)['logits']
            correct += (labels == torch.argmax(logits, axis=1)).sum()
            
    val_pd_acc = correct / val_len
    if model.is_real:
        wandb.log({'val_acc_real': val_pd_acc})
    else:
        wandb.log({'val_acc_pred': val_pd_acc})