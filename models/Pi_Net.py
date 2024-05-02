import torch.nn as nn


class Pi_Net(nn.Module):
    def __init__(self, in_channels=3):
        super(Pi_Net, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(),
            nn.AvgPool2d(4)
        )
        self.dense = nn.Sequential(
            nn.Linear(1024, 2500),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=in_channels, kernel_size=51, padding=25),
            nn.BatchNorm2d(num_features=in_channels),
            nn.Sigmoid()
        )

    def forward(self, batch):
        x = batch['items']
        bs = x.shape[0]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.dense(x)
        x = x.view(-1, 1, 50, 50)
        x = self.decoder(x)
        return {
            'pred_pis': x.view(bs, -1)
        }
