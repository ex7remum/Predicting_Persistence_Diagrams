import torch.nn as nn
from torch.nn import CrossEntropyLoss


class CustomCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.ce = CrossEntropyLoss(reduction=reduction)

    def forward(self, batch, model_out):
        return self.ce(model_out['logits'], batch['labels'])
