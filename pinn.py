import math
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(in_features, is_first)

    def _init_weights(self, in_features, is_first):
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_features
            else:
                bound = math.sqrt(6.0 / in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    def __init__(self, hidden_layers=4, hidden_width=64, omega_0=30):
        super().__init__()
        layers = [SineLayer(2, hidden_width, is_first=True, omega_0=omega_0)]
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_width, hidden_width, is_first=False, omega_0=omega_0))
        self.hidden = nn.Sequential(*layers)

        self.output = nn.Linear(hidden_width, 1)
        bound = math.sqrt(6.0 / hidden_width) / omega_0
        with torch.no_grad():
            self.output.weight.uniform_(-bound, bound)
            self.output.bias.uniform_(-bound, bound)

    def forward(self, xy):
        return self.output(self.hidden(xy))


class PINN(nn.Module):
    def __init__(self, hidden_layers=4, hidden_width=64):
        super().__init__()

        layers = [nn.Linear(2, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
        layers += [nn.Linear(hidden_width, 1)]

        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        return self.net(xy)

