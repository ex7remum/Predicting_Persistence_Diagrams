import torch.nn as nn
from torch.nn import MSELoss


class PIMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = MSELoss(reduction=reduction)

    def forward(self, batch, model_out):
        return self.mse(batch['pis'], model_out['pred_pis'])
