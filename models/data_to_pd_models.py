import torch
import torch.nn as nn
from torch import Tensor
import torchvision
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class OneShotPd(nn.Module):
    def __init__(self, encoder_data: nn.Module, decoder_pd: nn.Module, size_predictor : nn.Module, n_max : int):
        super(OneShotPd, self).__init__()
        self.encoder_data = encoder_data
        self.decoder_pd = decoder_pd
        self.size_predictor = size_predictor
        self.output = nn.Softplus()
        self.n_max = n_max

    def forward(self, X: Tensor, mask_pd = None):
        if mask_pd is None:
            batch_size = X.shape[0]
            self.size_predictor.eval()
            with torch.no_grad():
                sizes = self.size_predictor(X).squeeze(-1) 
                sizes = torch.ceil(sizes)
                for i in range(len(sizes)):
                    sizes[i] = max(1, sizes[i])
                    sizes[i] = min(self.n_max, sizes[i])
                    
                mask_pd = torch.zeros((batch_size, int(torch.max(sizes).item()))).to(torch.long).to(device)
                
                for i in range(len(sizes)):
                    mask_pd[i, :int(sizes[i])] = 1
                
            
        z_enc = self.encoder_data(X)
        
        z = self.decoder_pd(z_enc, mask_pd.shape[1], mask_pd)
        res = self.output(z) # to be non-negative
        return res * mask_pd.unsqueeze(2)
    
# TODO: add DETR