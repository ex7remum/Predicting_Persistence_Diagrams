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
import trainer

def get_dataloaders(path_to_config):
    f = open(path_to_config)
    config = json.load(f)
    train_dataset = getattr(datasets, config['data']['train']['dataset']['type'])(**config['data']['train']['dataset']['args'])
    
    generator = torch.Generator().manual_seed(42)
    train1_dataset, train2_dataset = torch.utils.random_split(train_dataset, [0.5, 0.5], generator=generator)
    
    test_dataset = getattr(datasets, config['data']['test']['dataset']['type'])(**config['data']['test']['dataset']['args'])
    
    collator = getattr(collate_fn, config['collator']['type'])

    trainloader1 = DataLoader(train_dataset1, batch_size=config['data']['train']['batch_size'], 
                             num_workers=config['data']['train']['num_workers'], shuffle=True, drop_last=True, collate_fn=collator)
    
    trainloader2 = DataLoader(train_dataset2, batch_size=config['data']['train']['batch_size'], 
                             num_workers=config['data']['train']['num_workers'], shuffle=True, drop_last=True, collate_fn=collator)    
    
    testloader = DataLoader(test_dataset, batch_size=config['data']['test']['batch_size'], 
                            num_workers=config['data']['test']['num_workers'], shuffle=False, collate_fn=collator)
    return trainloader1, trainloader2, testloader


def get_train_model_params(path_to_config):
    f = open(path_to_config)
    config = json.load(f)
    model = getattr(models, config['arch']['type'])(**config['arch']['args'])
    
    optimizer = getattr(torch.optim, config['optimizer']['type'])(model.parameters(), **config['optimizer']['args'])
    
    if 'lr_scheduler' in config:
        scheduler = getattr(torch.optim.lr_scheduler, config['lr_scheduler']['type'])(optimizer, **config['lr_scheduler']['args'])
    else:
        scheduler = None
        
    return model, optimizer, scheduler


def train_class_model(path_to_config, is_real, model_pd, trainloader, testloader):
    f = open(path_to_config)
    config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, scheduler = get_train_model_params(path_to_config)
    model = model.to(device)
    final_model = trainer.train_loop_class(model, trainloader, testloader, optimizer, is_real, device, model_pd
                             scheduler, n_epochs=config["trainer"]["n_epochs"], clip_norm=config["trainer"]["grad_norm_clip"])
    return final_model


def run_exp_full(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader1, trainloader2, testloader = get_dataloaders(args.config)
    model, optimizer, scheduler = get_train_model_params(args.config)
    model = model.to(device)
    
    f = open(args.config)
    config = json.load(f)
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
    final_model = trainer.train_loop_pd(model, trainloader1, testloader, optimizer, loss_fn, device, 
                             scheduler, n_epochs=config["trainer"]["n_epochs"], clip_norm=config["trainer"]["grad_norm_clip"])
    
    os.makedirs('pretrained_models', exist_ok=True)
    torch.save(final_model.state_dict(), f'pretrained_models/{run}_model.pth')
    
    metrics = utils.get_metrics(trainloader2, testloader, 'pd', final_model, pimgr)
    wandb.log(metrics)
    
    class_model_real = train_class_model(args.class_config, True, None, trainloader2, testloader)
    class_model_pred = train_class_model(args.class_config, False, final_model, trainloader2, testloader)
    
    torch.save(class_model_real.state_dict(), f'pretrained_models/{run}_class_real_model.pth')
    torch.save(class_model_pred.state_dict(), f'pretrained_models/{run}_class_pred_model.pth')
    
    acc_real = metrics.calculate_accuracy_on_pd(None, class_model_real, testloader, True)
    acc_pred = metrics.calculate_accuracy_on_pd(final_model, class_model_pred, testloader, False)
    
    wandb.log({'acc_real_pd': acc_real, 'acc_pred_pd': acc_pred})
    
    wandb.finish()
    

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
    
    parser.add_argument(
        "-l",
        "--class_config",
        default=None,
        type=str,
        help="path to classification model config",
    )
    
    args = parser.parse_args()
    run_exp_full(args)