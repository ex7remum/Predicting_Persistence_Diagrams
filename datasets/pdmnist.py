import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import pickle as pkl
from datasets.base_dataset import BasePDDataset


class PDMnist(BasePDDataset):
    def __init__(self, data_dir, pd_dir, train, limit=None, leave=None):
        self.leave = leave
        
        transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,))])
        self.dataset = torchvision.datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
        
        with open(pd_dir, 'rb') as f:
            self.pds = pkl.load(f)
            
        if limit is not None:
            self.pds = self.pds[:limit]
            self.dataset = self.dataset[:limit]