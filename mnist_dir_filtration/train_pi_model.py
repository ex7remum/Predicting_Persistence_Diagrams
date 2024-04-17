import torch
import wandb
from torch.utils.data import DataLoader
import argparse
import json
import datasets
import utils
import models
import collate_fn
import os
from torch.nn import MSELoss, BCELoss
from gudhi.representations.vector_methods import PersistenceImage as PersistenceImageGudhi
import torchvision


def val_step(model, valloader, pimgr, device):
    model.eval()
    pie = 0.
    mse = MSELoss(reduction='sum')
    val_len = len(valloader.dataset)
    
    log_img = True
    
    for item in valloader:
        X, Z, v = item['items'], item['pds'], item['labels']
        with torch.no_grad():
            Z = Z[..., :2].to(torch.float32)
            PI_pred = model(X.to(device))
            PI_real = torch.from_numpy(pimgr.fit_transform(Z)).to(torch.float32).to(device)
            PI_real = PI_real / PI_real.max(dim=1, keepdim=True)[0]
            pie += mse(PI_pred, PI_real)
            
        if log_img:
            PI_real = PI_real.view(-1, 50, 50)
            PI_pred = PI_pred.view(-1, 50, 50)
            
            PI_real_grid = torchvision.utils.make_grid(PI_real)
            PI_pred_grid = torchvision.utils.make_grid(PI_pred)
            
            images_real = wandb.Image(PI_real_grid, caption="Real PI")
            images_pred = wandb.Image(PI_pred_grid, caption="Pred PI")
            
            wandb.log({"Real PI": images_real})
            wandb.log({"Pred PI": images_pred})
            
            # log only first batch
            log_img = False

    wandb.log({'val_pie': pie / val_len})
    

def train_loop(model, trainloader, valloader, optimizer, pimgr, device, scheduler=None, n_epochs=25, clip_norm=None):
    torch.manual_seed(0)
    loss_fn = BCELoss()
    
    for _ in range(n_epochs):
        model.train()
        for item in trainloader:
            X, Z, v = item['items'], item['pds'], item['labels']
            optimizer.zero_grad()
            
            Z = Z[..., :2].to(torch.float32)
            PI_pred = model(X.to(device))
            PI_real = torch.from_numpy(pimgr.fit_transform(Z)).to(torch.float32).to(device)
            PI_real = PI_real / PI_real.max(dim=1, keepdim=True)[0]

            loss = loss_fn(PI_pred, PI_real)
            
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
        
        val_step(model, valloader, pimgr, device)
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

    final_model = train_loop(model, trainloader, testloader, optimizer, pimgr, device, 
                             scheduler, n_epochs=config["trainer"]["n_epochs"], clip_norm=config["trainer"]["grad_norm_clip"])
    
    os.makedirs('pretrained_models', exist_ok=True)
    torch.save(final_model.state_dict(), f'pretrained_models/{run}_model.pth')
    wandb.finish()