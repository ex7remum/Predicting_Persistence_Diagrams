import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn.functional import relu
import torch.nn.functional as F
from torch import Tensor
import math


# Transformer classes
class Attention(nn.Module):
    # Single-head attention
    def __init__(self, embed_dim, num_heads, dropout = 0.1):
        super().__init__()
        attention_dim = embed_dim // num_heads

        self.WQ = nn.Linear(embed_dim, attention_dim, bias=False)
        self.WK = nn.Linear(embed_dim, attention_dim, bias=False)
        self.WV = nn.Linear(embed_dim, attention_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, length, embed_dim)
        # mask: (batch_size, length, length)

        Q = self.WQ(query)
        K = self.WK(key)
        V = self.WV(value)
        # Q, K, V: (batch_size, length, attention_dim)

        norm_factor = math.sqrt(Q.shape[-1])
        dot_products = torch.bmm(Q, K.transpose(1, 2)) / norm_factor
        # dot_products: (batch_size, length, length)

        if mask is not None:
            dot_products = dot_products.masked_fill(mask, -math.inf)

        attention_score = nn.functional.softmax(dot_products, dim=-1)
        attention = torch.bmm(self.dropout(attention_score), V)
        # attention_score: (batch_size, length, length)
        # attention: (batch_size, length, attention_dim)

        return attention, attention_score

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()

        assert embed_dim % num_heads == 0
        self.attention_heads = nn.ModuleList([Attention(embed_dim, num_heads, dropout)
                                              for _ in range(num_heads)])

        self.linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, length, embed_dim)
        # mask: (batch_size, length, length)
        attentions, attention_scores = [], []

        for head in self.attention_heads:
            attention, attention_score = head(query, key, value, mask)
            attentions += [attention]
            attention_scores += [attention_score]

        attentions = torch.cat(attentions, dim=-1)
        attention_scores = torch.stack(attention_scores, dim=-1)
        # attentions: (batch_size, length, embed_dim)
        # attention_scores: (batch_size, length, length, num_heads)

        outputs = self.linear(attentions)
        outputs = self.dropout(outputs)
        # outputs: (batch_size, length, embed_dim)

        return outputs, attention_scores

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, fc_dim, num_heads, dropout = 0.0, activation = 'relu', use_skip = True):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.skip = use_skip
        if activation == 'relu':
            self.feedforward = nn.Sequential(
                nn.Linear(embed_dim, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_dim, embed_dim),
                nn.Dropout(dropout)
            )
        elif activation == 'gelu':
            self.feedforward = nn.Sequential(
                nn.Linear(embed_dim, fc_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fc_dim, embed_dim),
                nn.Dropout(dropout)
            )
        else:
            raise NotImplementedError
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, inputs, mask = None):
        attention, attention_score = self.self_attention(query=inputs, key=inputs,
                                                         value=inputs, mask=mask)
        outputs = inputs + attention
        outputs = self.norm1(outputs)
        outputs = outputs + self.feedforward(outputs)
        outputs = self.norm2(outputs)
        if self.skip:
            return outputs + inputs, attention_score
        else:
            return outputs, attention_score

def create_padding_mask(mask: Tensor):
    # tokens: (batch_size, length)
    length = mask.shape[-1]
    padding_mask = (mask == 0)
    padding_mask = padding_mask.unsqueeze(1).repeat(1, length, 1)
    # padding_mask: (batch_size, length, length)

    return padding_mask