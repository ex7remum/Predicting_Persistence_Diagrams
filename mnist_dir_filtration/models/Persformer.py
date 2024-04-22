import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn.functional import relu
import torch.nn.functional as F
from torch import Tensor
import math
import models
import numpy as np


# Persformer from https://arxiv.org/abs/2112.15210
class CustomPersformer(nn.Module):
    def __init__(self, n_in, embed_dim, fc_dim, num_heads, num_layers, n_out_enc,
                 dropout = 0.1, reduction="mean", use_skip=True):
        super(CustomPersformer, self).__init__()
        self.embed_dim = embed_dim
        self.reduction = reduction
        self.embedding = nn.Sequential(
            nn.Linear(n_in, embed_dim),
            nn.GELU(),
        )
        # (batch, length, 3) -> (batch, length, emb_dim)

        self.query = nn.parameter.Parameter(
            torch.Tensor(1, embed_dim), requires_grad=True
        )
        self.scaled_dot_product_attention = models.MultiHeadAttention(embed_dim, num_heads, dropout)

        self.layers = nn.ModuleList([models.TransformerLayer(embed_dim, fc_dim, num_heads, dropout, 'gelu', use_skip) \
                                     for _ in range(num_layers)])

        self.output = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_out_enc),
        )

    def forward(self, X, mask=None):
        if mask is None:
            mask = torch.ones_like(X).to(X.device)[..., 0]
        outputs = self.embedding(X) * math.sqrt(self.embed_dim)
        padding_mask = models.create_padding_mask(mask)
        for layer in self.layers:
            outputs, attention_score = layer(outputs, padding_mask)

        if self.reduction == "mean":
            lengths = mask.sum(dim=1).detach()
            outputs = (outputs * mask.unsqueeze(2)).sum(dim=1) / lengths.unsqueeze(1)
        elif self.reduction == "attention":
            outputs, _ = self.scaled_dot_product_attention(
                self.query.expand(outputs.shape[0], -1, -1),
                outputs,
                outputs,
                mask = padding_mask[:, 0, :].unsqueeze(1),
            )
            outputs = outputs.squeeze(dim=1)
        else:
            raise NotImplementedError

        # outputs : (batch_size, emb_dim)

        z = self.output(outputs)
        # z : (batch_size, n_out_enc)

        return z