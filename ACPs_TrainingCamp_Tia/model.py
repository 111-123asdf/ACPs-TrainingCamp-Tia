import torch.nn as nn

class FlexibleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim,
                 activation='relu', dropout_rate=0.0, use_bn=True):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU(0.1))
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
