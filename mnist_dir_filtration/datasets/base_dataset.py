import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import pickle as pkl

class BasePDDataset(Dataset):
    def __init__(self, data_dir, pd_dir, limit=None):
        with open(pd_dir, 'rb') as f:
            self.pds = pkl.load(f)
        with open(data_dir, 'rb') as f:
            self.dataset = pkl.load(f)
            
        if limit is not None:
            self.pds = self.pds[:limit]
            self.dataset = self.dataset[:limit]
        
    def __len__(self):
        return len(self.pds)
    
    def __getitem__(self, idx):
        item, label = self.dataset[idx]
        pd = self.pds[idx]
        return item, pd, label