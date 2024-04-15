import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.functional import relu
from .layers import MLP

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PI_Net(nn.Module):
    def __init__(self, in_channels = 3):
        super(PI_Net, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = 128, kernel_size = 3, stride = 1, padding = 2),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 2),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 2), 
            nn.BatchNorm2d(num_features = 512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2) 
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 1, padding = 2), 
            nn.BatchNorm2d(num_features = 1024),
            nn.ReLU(),
            nn.AvgPool2d(4)
        )
        self.dense = nn.Sequential(
            nn.Linear(1024, 2500),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 1, out_channels = in_channels, kernel_size = 51, padding = 25),
            nn.BatchNorm2d(num_features = in_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.dense(x)
        x = x.reshape((-1, 1, 50, 50))
        x = self.decoder(x)
        return x.reshape((-1, 50 * 50 * x.shape[1]))
    
    
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) 
                                         #  batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature
    
    
class TopologyNet(nn.Module):
    def __init__(self, k=10, in_channels=3, dropout=0.1):
        super(TopologyNet, self).__init__()
        self.k = k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(448, 1024, bias=False)
        self.linear2 = nn.Linear(1024, 2500, bias=False)
        
        self.dp1 = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = x.transpose(2, 1)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)
        
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        
        x = F.leaky_relu(self.linear1(x), negative_slope=0.2)
        x = self.dp1(x)
        x = self.linear2(x)
        return x
    
class RipsNet(nn.Module):
    def __init__(self, in_channels=3):
        super(RipsNet, self).__init__()
        # self.linear1 = nn.Linear(in_channels, 64)
        # self.linear2 = nn.Linear(64, 64)
        # self.linear3 = nn.Linear(64, 64)
        # self.linear4 = nn.Linear(64, 64)
        # self.linear5 = nn.Linear(64, 100)
        # self.linear6 = nn.Linear(100, 200)
        # self.linear7 = nn.Linear(200, 2500)
        self.MLP1 = MLP(dim_in=in_channels, dim_out=64, width=64, nb_layers=3,
                               skip=0)
        self.MLP2 = MLP(dim_in=64, dim_out=2500, width=256, nb_layers=1,
                               skip=0)
        
    def forward(self, x):
#         x = self.linear1(x)
#         x = relu(x)
#         x = self.linear2(x)
#         x = relu(x)
#         x = self.linear3(x)
#         x = relu(x)
#         x = x.sum(dim=1)
        
#         x = self.linear4(x)
#         x = relu(x)
#         x = self.linear5(x)
#         x = relu(x)
#         x = self.linear6(x)
#         x = relu(x)
#         x = self.linear7(x)
        x = self.MLP1(x)
        
        x = x.mean(dim=1)
        
        x = self.MLP2(x)
        
        return x
    