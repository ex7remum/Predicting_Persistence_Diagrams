import torch
import datasets
import models
import losses
from gudhi.representations.vector_methods import PersistenceImage as PersistenceImageGudhi


def get_dataloaders(config):
    train_dataset = getattr(datasets, config['data']['train']['dataset']['type'])(
        **config['data']['train']['dataset']['args'])

    generator = torch.Generator().manual_seed(42)
    train1_dataset, train2_dataset = torch.utils.data.dataset.random_split(train_dataset, [0.5, 0.5],
                                                                           generator=generator)

    test_dataset = getattr(datasets, config['data']['test']['dataset']['type'])(
        **config['data']['test']['dataset']['args'])

    return train1_dataset, train2_dataset, test_dataset


def get_train_model_params(config):
    model = getattr(models, config['arch']['type'])(**config['arch']['args'])

    optimizer = getattr(torch.optim, config['optimizer']['type'])(model.parameters(), **config['optimizer']['args'])

    if 'lr_scheduler' in config:
        scheduler = getattr(torch.optim.lr_scheduler, config['lr_scheduler']['type'])(optimizer,
                                                                                      **config['lr_scheduler']['args'])
    else:
        scheduler = None

    return model, optimizer, scheduler


def get_loss_fn(config):
    loss_fn = getattr(losses, config['loss']['type'])(**config['loss']['args'])
    return loss_fn


def get_pimgr(config):
    if 'pimgr' in config:
        pimgr = PersistenceImageGudhi(resolution=[50, 50],
                                      weight=lambda x: x[1],
                                      **config['pimgr'])
    else:
        pimgr = None

    return pimgr
