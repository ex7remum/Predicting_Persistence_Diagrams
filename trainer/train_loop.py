import torch
import wandb
from trainer import move_batch_to_device


def train_loop(model, trainloader, valloader, optimizer, loss_fn, device, val_function,
               scheduler=None, n_epochs=25, clip_norm=None, seed=0):
    torch.manual_seed(seed)

    for _ in range(n_epochs):
        model.train()
        for batch in trainloader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(batch, out)

            loss.backward()

            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_norm = 0
            parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            for p in parameters:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            lr = optimizer.param_groups[0]['lr']
            wandb.log({f'loss_{type}': loss,
                       f'grad_norm_{type}': total_norm,
                       f'learning_rate_{type}': lr})

        val_function(model, valloader, device)
    return model
