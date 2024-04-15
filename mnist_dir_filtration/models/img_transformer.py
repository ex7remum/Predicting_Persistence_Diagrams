import torch
import torch.nn as nn
from torch.nn import Module, Transformer, Linear, Parameter
from torch.nn.functional import softmax, relu
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR



# https://github.com/s-chh/PyTorch-Vision-Transformer-ViT-MNIST-CIFAR10/blob/main/model.py
class EmbedLayer(nn.Module):
    def __init__(self, n_channels, embed_dim, image_size, patch_size):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)  # Pixel Encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, embed_dim), requires_grad=True)  # Positional Embedding

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)  # B C IH IW -> B E IH/P IW/P (Embedding the patches)
        x = x.reshape([x.shape[0], x.shape[1], -1])  # B E IH/P IW/P -> B E S (Flattening the patches)
        x = x.transpose(1, 2)  # B E S -> B S E 
        x = x + self.pos_embedding  # Adding positional embedding
        return x


class ImageSet2Set(nn.Module):
    
    def __init__(self, n_out_max=64, d_in=2, d_out=2, d_hidden=32, d_mlp=128, n_heads=4, n_encoder_layers=2, n_decoder_layers=2,
                 n_channels=1, image_size=28, patch_size=4):
        super().__init__()
        self.embed = EmbedLayer(n_channels, d_hidden, image_size, patch_size)
        self.linear_output = Linear(d_hidden, d_out)
        self.Q = Parameter(torch.rand(n_out_max, d_hidden))
        self.transformer = Transformer(d_hidden, n_heads, n_encoder_layers, n_decoder_layers, dim_feedforward=d_mlp, dropout=0.0, batch_first=True)
        
    def forward(self, X):
        batch_size = X.shape[0]
        X = self.embed(X)
        Z = self.transformer(X, self.Q.unsqueeze(0).repeat(batch_size, 1, 1))
        return self.linear_output(Z)