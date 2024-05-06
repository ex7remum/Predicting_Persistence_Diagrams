import torch
import torchvision
from torchvision import transforms
from tqdm.notebook import tqdm
import os
import pickle as pkl
import json
import utils
import argparse
from itertools import product
from sklearn.model_selection import train_test_split


def get_pds_from_data(dataset_type, data_path, filtration_path_name, limit=None, **kwargs):
    filtration_func = getattr(utils, kwargs['filtration_func']['type'])

    os.makedirs(filtration_path_name, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)

    # get dataset items and labels
    if dataset_type == "MNIST":
        dataset_train = torchvision.datasets.MNIST(root=data_path, train=True, download=True,
                                                   transform=transforms.ToTensor())
        dataset_test = torchvision.datasets.MNIST(root=data_path, train=False, download=True,
                                                  transform=transforms.ToTensor())

    elif dataset_type == "Orbit":
        dataset_args = kwargs['dataset_args']
        m_over, m, n, rr = dataset_args['m_over'], dataset_args['m'], dataset_args['n'], dataset_args['rr']

        if 'device' in dataset_args:
            device = dataset_args['device']
        else:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if 'seed' in dataset_args:
            random_seed = dataset_args['seed']
        else:
            random_seed = 0

        gen_orbits = utils.get_orbit_dataset(m_over, m, n, rr, random_seed, device)
        dataset = []

        for k, (i, j) in enumerate(tqdm(list(product(range(rr), range(m))))):
            dataset.append((gen_orbits[i, j], i))

        if 'test_size' in dataset_args:
            test_size = dataset_args['test_size']
        else:
            test_size = 0.3

        dataset_train, dataset_test = train_test_split(dataset, test_size=test_size)
        with open(f'{data_path}/orbit{m*rr}k_train.pkl', 'wb') as f:
            pkl.dump(dataset_train, f)

        with open(f'{data_path}/orbit{m*rr}k_test.pkl', 'wb') as f:
            pkl.dump(dataset_test, f)

    elif dataset_type == "Dynamic3D":
        dataset_args = kwargs['dataset_args']
        m_over, m, n, rr = dataset_args['m_over'], dataset_args['m'], dataset_args['n'], dataset_args['rr']
        gen_points = utils.generate_3D_dynamic(m_over=m_over, m=m, n=n)
        dataset = []
        for k, (i, j) in enumerate(tqdm(list(product(range(rr), range(m))))):
            dataset.append((gen_points[i, j], i))
        if 'test_size' in dataset_args:
            test_size = dataset_args['test_size']
        else:
            test_size = 0.3

        dataset_train, dataset_test = train_test_split(dataset, test_size=test_size)
        with open(f'{data_path}/dynamic3D_train.pkl', 'wb') as f:
            pkl.dump(dataset_train, f)

        with open(f'{data_path}/dynamic3D_test.pkl', 'wb') as f:
            pkl.dump(dataset_test, f)

    elif dataset_type == "Obayashi-Hiraoka":
        dataset_args = kwargs['dataset_args']
        n_images = dataset_args['n_images']
        gen_images = utils.generate_ob_hir(**dataset_args)
        gen_images = torch.from_numpy(gen_images).to(torch.float32).unsqueeze(1)
        labels = torch.ones(n_images)
        labels[:n_images//2] = 0
        dataset = []
        for image, label in zip(gen_images, labels):
            dataset.append((image, label))

        if 'test_size' in dataset_args:
            test_size = dataset_args['test_size']
        else:
            test_size = 0.3
        dataset_train, dataset_test = train_test_split(dataset, test_size=test_size)
        with open(f'{data_path}/ob_hir_train.pkl', 'wb') as f:
            pkl.dump(dataset_train, f)

        with open(f'{data_path}/ob_hir_test.pkl', 'wb') as f:
            pkl.dump(dataset_test, f)

    # generate PDs
    pds_train = []
    for i, (obj, label) in tqdm(enumerate(dataset_train)):
        diags = filtration_func(obj, **kwargs['filtration_func']['args'])
        pds_train.append(diags)

        if limit is not None and len(pds_train) >= limit:
            break

    name = f'{dataset_type}_pds.pkl'

    with open(filtration_path_name + '/' + name, 'wb') as f:
        pkl.dump(pds_train, f)

    if dataset_test is not None:
        pds_test = []
        for i, (obj, label) in tqdm(enumerate(dataset_test)):
            diags = filtration_func(obj, **kwargs['filtration_func']['args'])
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
    get_pds_from_data(config['dataset_type'], config['data_path'],
                      config['filtration_path_name'], config['limit'], **config['setup'])
