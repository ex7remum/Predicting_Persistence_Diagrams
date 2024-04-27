import torch
import torch.nn as nn
from models import TransformerLayer, MLP, create_padding_mask
import numpy as np


def _get_full_mask(X, mask):
    full_mask = np.zeros(X.shape)
    for i in range(X.shape[0]):
        full_mask[i, :mask[i].sum()] = np.ones(X.shape[-1])

    full_mask = torch.tensor(full_mask).long().to(X.device)
    return full_mask


class TransformerDecoder(nn.Module):
    def __init__(self, n_in, latent_dim, fc_dim, num_heads, num_layers, n_out, generator,
                 n_out_lin, n_hidden, num_layers_lin, dropout=0.1, use_conv=False, last_mlp_width=32,
                 last_mlp_layers=2):
        super(TransformerDecoder, self).__init__()
        self.set_generator = generator

        self.film_mlp = MLP(dim_in=n_in, width=n_hidden, dim_out=n_out_lin, nb_layers=num_layers_lin,
                            dim_in_2=latent_dim, modulation='film')

        self.layers = nn.ModuleList([TransformerLayer(n_out_lin, fc_dim, num_heads, dropout) \
                                     for _ in range(num_layers)])

        self.use_conv = use_conv
        if use_conv:
            self.output = nn.Conv1d(n_out_lin, n_out, 1)
        else:
            self.output = MLP(dim_in=n_out_lin, width=last_mlp_width, dim_out=n_out, nb_layers=last_mlp_layers)

    def forward(self, latent, n_max, mask):
        ref_set = self.set_generator(latent, n_max)

        full_mask = _get_full_mask(ref_set, mask)

        # concat version
        # outputs = torch.cat((ref_set, latent.unsqueeze(1).repeat(1, n_max, 1)), 2)
        # (batch_size, n_max_batch, set_channels + latent_dim)
        # outputs_full_mask = get_full_mask(outputs, mask)
        # outputs = outputs * outputs_full_mask + (1 - outputs_full_mask) * (-1)

        # film version
        outputs = self.film_mlp(ref_set, latent.unsqueeze(1).repeat(1, n_max, 1))
        # (batch_size, n_max_batch, n_out)

        # attention_scores = []
        padding_mask = create_padding_mask(full_mask[:, :, 0])
        for layer in self.layers:
            outputs, attention_score = layer(outputs, padding_mask)

        if self.use_conv:
            z = self.output(outputs.transpose(1, 2)).transpose(1, 2)
        else:
            z = self.output(outputs)
        # z : (batch_size, n_max_batch, set_channels)

        return z * full_mask


class MLPDecoder(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, num_layers, generator):
        super(MLPDecoder, self).__init__()
        self.mlp = MLP(dim_in=n_in, width=n_hidden, dim_out=n_out, nb_layers=num_layers)
        self.generator = generator

    def forward(self, latent, max_batch_len, mask):
        latent = self.generator(latent, max_batch_len)
        # (bs, n_batch, n_in)
        z = self.mlp(latent)
        full_mask = _get_full_mask(z, mask)
        return full_mask * z
