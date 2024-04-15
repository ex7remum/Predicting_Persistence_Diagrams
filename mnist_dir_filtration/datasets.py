import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import math
import numpy as np
from utils import process_image
from tqdm.notebook import tqdm

class PDMnist(Dataset):
    def __init__(self, data_dir='data', train=True, num_filtrations=4, limit=None, leave=None):
        self.dataset = torchvision.datasets.MNIST(root=data_dir, train=train, download=True, transform=transforms.ToTensor())
        self.pds = []
        self.idxs = []
        filter_params = np.arange(num_filtrations) / num_filtrations * 2 * math.pi
        for i, (img, label) in tqdm(enumerate(self.dataset)):
            diags, _ = process_image(img, filter_params)
            
            if leave is not None:
                if len(diags) >= leave:
                
                    lifetime = diags[:, 1] - diags[:, 0]
                    order = torch.argsort(lifetime, descending=True)
                    diags = diags[order][:leave]
                    self.pds.append(diags)
                    self.idxs.append(i)
            else:
                self.pds.append(diags)
                self.idxs.append(i)
            
            if limit is not None and len(self.pds) >= limit:
                break
        
    def __len__(self):
        return len(self.pds)
    
    def __getitem__(self, idx):
        real_idx = self.idxs[idx]
        mnist_img, mnist_label = self.dataset[real_idx]
        pd = self.pds[idx]
        return mnist_img, pd, mnist_label