import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import math
import numpy as np
from tqdm.notebook import tqdm
import os
import pickle as pkl
import json
import utils
import argparse


def get_pds_from_data(dataset_type, data_path, filtration_func, filtration_path_name, limit=None, **kwargs):
    # kwargs - additional params to filtration function
    
    os.makedirs(filtration_path_name, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)
    
    if dataset_type == "MNIST":
        dataset_train = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        dataset_test = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transforms.ToTensor())
    
    pds_train = []
    for i, (img, label) in tqdm(enumerate(dataset_train)):
        diags, _ = filtration_func(img, **kwargs)
        pds_train.append(diags)
        
        if limit is not None and len(pds_train) >= limit:
            break
    
    name = f'{dataset_type}_pds.pkl'
    
    with open(filtration_path_name + '/' + name, 'wb') as f:
        pkl.dump(pds_train, f)
        
    if dataset_test is not None:
        pds_test = []
        for i, (img, label) in tqdm(enumerate(dataset_test)):
            diags, _ = filtration_func(img, **kwargs)
            pds_test.append(diags)
            
            if limit is not None and len(pds_test) >= limit:
                break
        name = f'{dataset_type}_pds_test.pkl'
        with open(filtration_path_name + '/' + name, 'wb') as f:
            pkl.dump(pds_test, f)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="path to dataset config",
    )
    args = parser.parse_args()
    
    f = open(args.config)
    config = json.load(f)
    filt_fn = getattr(utils, config['filtration_func']['type'])
    get_pds_from_data(config['dataset_type'], config['data_path'], 
                  filt_fn, config['filtration_path_name'], config['limit'], **config['filtration_func']['args'])