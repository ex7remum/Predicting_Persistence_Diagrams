import torch
import torch.nn as nn
from torch.nn import Module, Transformer, Linear, Parameter


class Set2Set(nn.Module):
    def __init__(self, n_out_max=64, d_in=2, d_out=2, d_hidden=32, d_mlp=128, n_heads=4,
                 n_encoder_layers=2, n_decoder_layers=2):
        super().__init__()
        self.linear_input = Linear(d_in, d_hidden)
        self.linear_output = Linear(d_hidden, d_out)
        self.Q = Parameter(torch.rand(n_out_max, d_hidden))
        self.transformer = Transformer(d_hidden, n_heads, n_encoder_layers, n_decoder_layers, 
                                       dim_feedforward=d_mlp, dropout=0.0, batch_first=True)
        self.output = nn.Softplus()

    def forward(self, batch):
        X = batch['items']
        batch_size = X.shape[0]
        X = self.linear_input(X)
        Z = self.transformer(X, self.Q.unsqueeze(0).repeat(batch_size, 1, 1))
        out = self.linear_output(Z)
        return {
            'pred_pds': self.output(out)
        }
