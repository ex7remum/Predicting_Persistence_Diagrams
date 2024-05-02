import torch.nn as nn
from torch.nn import BCELoss


class PIBCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.bce = BCELoss(reduction=reduction)

    def forward(self, batch, model_out):
        return self.bce(model_out['pred_pis'], batch['pis'])
