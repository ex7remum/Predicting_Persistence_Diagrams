import torch
import wandb
from torch.utils.data import DataLoader
import argparse
import json
import datasets
import utils
import models
import losses
import collate_fn
import os
from gudhi.representations.vector_methods import PersistenceImage as PersistenceImageGudhi

def val_step(model, valloader, device):
    model.eval()
    metric_chamfer = 0.
    metric_hausdorff = 0.
    val_len = len(valloader.dataset)
    for item in valloader:
        X, Z, v = item['items'], item['pds'], item['labels']
        with torch.no_grad():
            Z = Z[..., :2].to(torch.float32).to(device)
            Z_hat = model(X.to(device))
        
        metric_chamfer += losses.ChamferLoss(reduce='sum')(Z_hat, Z)
        metric_hausdorff += losses.HausdorffLoss(reduce='sum')(Z_hat, Z)
    wandb.log({'val_chamfer': metric_chamfer / val_len, 'val_hausdorff': metric_hausdorff / val_len})

def train_loop(model, trainloader, valloader, optimizer, loss_fn, device, scheduler=None, n_epochs=25, clip_norm=None):
    torch.manual_seed(0)
    
    for _ in range(n_epochs):
        model.train()
        for item in trainloader:
            X, Z, v = item['items'], item['pds'], item['labels']
            optimizer.zero_grad()
            
            Z = Z[..., :2].to(torch.float32).to(device)
            Z_hat = model(X.to(device))
            loss = loss_fn(Z_hat, Z)
            
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
            wandb.log({'loss': loss, 'grad_norm': total_norm, 'learning rate': lr})
        
        val_step(model, valloader, device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-w",
        "--wandb_key",
        default=None,
        type=str,
        help="wandb key for logging",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="path to trainer config",
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    f = open(args.config)
    config = json.load(f)
    
    train_dataset = getattr(datasets, config['data']['train']['dataset']['type'])(**config['data']['train']['dataset']['args'])
    test_dataset = getattr(datasets, config['data']['test']['dataset']['type'])(**config['data']['test']['dataset']['args'])
    
    collator = getattr(collate_fn, config['collator']['type'])

    trainloader = DataLoader(train_dataset, batch_size=config['data']['train']['batch_size'], 
                             num_workers=config['data']['train']['num_workers'], shuffle=True, drop_last=True, collate_fn=collator)
    testloader = DataLoader(test_dataset, batch_size=config['data']['test']['batch_size'], 
                            num_workers=config['data']['test']['num_workers'], shuffle=False, collate_fn=collator)

    
    model = getattr(models, config['arch']['type'])(**config['arch']['args']).to(device)
    
    optimizer = getattr(torch.optim, config['optimizer']['type'])(model.parameters(), **config['optimizer']['args'])
    
    if 'lr_scheduler' in config:
        scheduler = getattr(torch.optim.lr_scheduler, config['lr_scheduler']['type'])(optimizer, **config['lr_scheduler']['args'])
    else:
        scheduler = None

    loss_fn = getattr(losses, config['loss']['type'])(**config['loss']['args'])
    
    
    if 'pimgr' in config:
        pimgr = PersistenceImageGudhi(resolution=[50, 50],
                                      weight=lambda x: x[1],
                                      **config['pimgr'])
    else:
        sigma, im_range = utils.compute_pimgr_parameters(train_dataset.pds)
        pimgr = PersistenceImageGudhi(bandwidth=sigma,
                                      resolution=[50, 50],
                                      weight=lambda x: x[1],
                                      im_range=im_range)
        

    run = config["trainer"]["run_name"]
    wandb.login(key=args.wandb_key)
    wandb.init(project=config["trainer"]["wandb_project"], 
               name=f"experiment_{run}",
               config=config
    )
    if 'pimgr' not in config:
        wandb.log({'sigma': sigma, 'min_b': im_range[0], 'max_b': im_range[1], 'min_p': im_range[2], 'max_p': im_range[3]})
        
    wandb.watch(model)

    final_model = train_loop(model, trainloader, testloader, optimizer, loss_fn, device, 
                             scheduler, n_epochs=config["trainer"]["n_epochs"], clip_norm=config["trainer"]["grad_norm_clip"])
    
    os.makedirs('pretrained_models', exist_ok=True)
    torch.save(final_model.state_dict(), f'pretrained_models/{run}_model.pth')
    
    metrics = utils.get_metrics(dataloader_train, dataloader_test, 'pd', final_model, pimgr)
    wandb.log(metrics)
    
    wandb.finish()