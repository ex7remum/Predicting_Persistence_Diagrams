import torch
import wandb
from torch.utils.data import DataLoader
import argparse
import json
import datasets
import models
import collate_fn
import os
import trainer


def get_dataloaders(path_to_config):
    f = open(path_to_config)
    config = json.load(f)
    train_dataset = getattr(datasets, config['data']['train']['dataset']['type'])(
        **config['data']['train']['dataset']['args'])

    generator = torch.Generator().manual_seed(42)
    train1_dataset, train2_dataset = torch.utils.data.dataset.random_split(train_dataset, [0.5, 0.5],
                                                                           generator=generator)

    test_dataset = getattr(datasets, config['data']['test']['dataset']['type'])(
        **config['data']['test']['dataset']['args'])

    collator = getattr(collate_fn, config['collator']['type'])

    trainloader1 = DataLoader(train1_dataset, batch_size=config['data']['train']['batch_size'],
                              num_workers=config['data']['train']['num_workers'], shuffle=True, drop_last=True,
                              collate_fn=collator)

    trainloader2 = DataLoader(train2_dataset, batch_size=config['data']['train']['batch_size'],
                              num_workers=config['data']['train']['num_workers'], shuffle=True, drop_last=True,
                              collate_fn=collator)

    testloader = DataLoader(test_dataset, batch_size=config['data']['test']['batch_size'],
                            num_workers=config['data']['test']['num_workers'], shuffle=False, collate_fn=collator)
    return train1_dataset, train2_dataset, trainloader1, trainloader2, testloader


def get_train_model_params(path_to_config):
    f = open(path_to_config)
    config = json.load(f)
    model = getattr(models, config['arch']['type'])(**config['arch']['args'])

    optimizer = getattr(torch.optim, config['optimizer']['type'])(model.parameters(), **config['optimizer']['args'])

    if 'lr_scheduler' in config:
        scheduler = getattr(torch.optim.lr_scheduler, config['lr_scheduler']['type'])(optimizer,
                                                                                      **config['lr_scheduler']['args'])
    else:
        scheduler = None

    return model, optimizer, scheduler


def run_exp_full(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train1_dataset, train2_dataset, trainloader1, trainloader2, testloader = get_dataloaders(args.config)

    f = open(args.config)
    config = json.load(f)

    model, optimizer, scheduler = get_train_model_params(args.config)
    model = model.to(device)

    run = config["trainer"]["run_name"]
    wandb.login(key=args.wandb_key, relogin=True)
    wandb.init(project=config["trainer"]["wandb_project"],
               name=f"experiment_{run}",
               config=config
               )

    wandb.watch(model)
    final_model = trainer.train_loop_size_pred(model, trainloader1, testloader, optimizer, device,
                                               scheduler, n_epochs=config["trainer"]["n_epochs"],
                                               clip_norm=config["trainer"]["grad_norm_clip"])

    os.makedirs('pretrained_models', exist_ok=True)
    torch.save(final_model.state_dict(), f'pretrained_models/{run}_model_size_pred.pth')
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

    args = parser.parse_args()
    run_exp_full(args)
