import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO : add ViT and similar maybe?

class Resnet18Encoder(nn.Module):
    def __init__(self, n_out_enc=256, pretrained=False, freeze=False):
        super().__init__()
        self.encoder = torchvision.models.resnet18(pretrained=pretrained)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.encoder.fc = nn.Linear(in_features=512, out_features=n_out_enc)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.encoder(x)
        return x


class SimpleCNNEncoder(nn.Module):
    def __init__(self, in_channels=1, n_out_enc=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_out_enc)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
