import torch.nn as nn
from torch.nn import MSELoss


class CustomMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.ce = MSELoss(reduction=reduction)

    def forward(self, batch, model_out):
        return self.ce(model_out['pred_size'], batch['lengths'])
