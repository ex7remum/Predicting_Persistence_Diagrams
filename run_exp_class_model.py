import torch
import wandb
from torch.utils.data import DataLoader
import argparse
import json
import datasets
import models
import collate_fn
import metrics
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


def train_class_model(path_to_config, is_real, model_pd, trainloader, testloader):
    f = open(path_to_config)
    config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, scheduler = get_train_model_params(path_to_config)
    model = model.to(device)
    final_model = trainer.train_loop_class(model, trainloader, testloader, optimizer, is_real, device, model_pd,
                                           scheduler, n_epochs=config["trainer"]["n_epochs"],
                                           clip_norm=config["trainer"]["grad_norm_clip"])
    return final_model


def run_exp_full(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train1_dataset, train2_dataset, trainloader1, trainloader2, testloader = get_dataloaders(args.config)

    f = open(args.config)
    config = json.load(f)

    run = config["trainer"]["run_name"]
    wandb.login(key=args.wandb_key, relogin=True)
    wandb.init(project=config["trainer"]["wandb_project"],
               name=f"experiment_{run}",
               config=config
               )

    os.makedirs('pretrained_models', exist_ok=True)
    if args.exp_type == 'real':
        class_model_real = train_class_model(args.class_config, True, None, trainloader2, testloader)
        torch.save(class_model_real.state_dict(), f'pretrained_models/{run}_class_real_model.pth')
        acc_real = metrics.calculate_accuracy_on_pd(None, class_model_real, testloader, True)
        wandb.log({'res_acc_real_pd_class': acc_real})

    else:
        pred_model = getattr(models, config['arch']['type'])(**config['arch']['args'])
        pred_model.load_state_dict(torch.load(args.model_path))
        pred_model = pred_model.to(device)
        class_model_pred = train_class_model(args.class_config, False, pred_model, trainloader2, testloader)
        torch.save(class_model_pred.state_dict(), f'pretrained_models/{run}_class_pred_model.pth')
        acc_pred = metrics.calculate_accuracy_on_pd(pred_model, class_model_pred, testloader, False)
        wandb.log({'res_acc_pred_pd_class': acc_pred})

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
    parser.add_argument(
        "-p",
        "--model_path",
        default=None,
        type=str,
        help="path to pretrained predictor model",
    )
    parser.add_argument(
        "-t",
        "--type",
        default=None,
        type=str,
        help="type ox experiment (real or pred)",
    )

    args = parser.parse_args()
    run_exp_full(args)
