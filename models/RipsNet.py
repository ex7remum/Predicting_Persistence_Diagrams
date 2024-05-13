import torch.nn as nn


class RipsNet(nn.Module):
    def __init__(self, in_channels=3, hidden_dims_first=[30, 20, 10], hidden_dims_second=[50, 100, 200],
                 dim_out=2500, activation='relu', agg='sum', use_sigmoid=True):
        super(RipsNet, self).__init__()
        self.layers_first = [nn.Linear(in_channels, hidden_dims_first[0])]
        for i in range(1, len(hidden_dims_first)):
            cur_lin = nn.Linear(hidden_dims_first[i - 1], hidden_dims_first[i])
            self.layers_first.append(cur_lin)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.layers_second = [nn.Linear(hidden_dims_first[-1], hidden_dims_second[0])]
        for i in range(1, len(hidden_dims_second)):
            cur_lin = nn.Linear(hidden_dims_second[i - 1], hidden_dims_second[i])
            self.layers_second.append(cur_lin)

        self.out = nn.Linear(hidden_dims_second[-1], dim_out)

        self.use_sigmoid = use_sigmoid
        self.agg = agg

    def forward(self, batch):
        x = batch['items']
        for layer in self.layers_first:
            x = layer(x)
            x = self.activation(x)

        if self.agg == 'sum':
            x = x.sum(dim=1)
        elif self.agg == 'mean':
            x = x.mean(dim=1)
        elif self.agg == 'max':
            x = x.max(dim=1)
        else:
            raise NotImplementedError

        for layer in self.layers_second:
            x = layer(x)
            x = self.activation(x)

        x = self.out(x)

        if self.use_sigmoid:
            x = x.sigmoid()

        return {
            'pred_pis': x
        }
