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


class SafeNet(nn.Module):
    def __init__(self, n_fourier=256, sigma=5.0, hidden_layers=1, hidden_width=64):
        super().__init__()
        B = torch.randn(n_fourier, 2) * sigma
        self.register_buffer('B', B)

        in_dim = 2 * n_fourier
        layers = [nn.Linear(in_dim, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
        layers += [nn.Linear(hidden_width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, xy):
        proj = xy @ self.B.T
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)
        return self.net(feat)


class HyResPINN(nn.Module):
    def __init__(self, hidden_layers=3, hidden_width=64, n_rbf=64):
        super().__init__()
        layers = [nn.Linear(2, hidden_width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_width, hidden_width), nn.Tanh()]
        layers += [nn.Linear(hidden_width, 1)]
        self.mlp = nn.Sequential(*layers)

        self.centers   = nn.Parameter(torch.rand(n_rbf, 2))
        self.log_sigma = nn.Parameter(torch.zeros(n_rbf))
        self.rbf_w     = nn.Parameter(torch.zeros(n_rbf))
        self.alpha_raw = nn.Parameter(torch.zeros(1))

    def _rbf(self, xy):
        diff  = xy.unsqueeze(1) - self.centers.unsqueeze(0)   # (N, K, 2)
        r2    = (diff ** 2).sum(-1)                            # (N, K)
        sigma2 = self.log_sigma.exp() ** 2                    # (K,)
        phi   = torch.exp(-r2 / (sigma2 + 1e-8))              # (N, K)
        return (phi * self.rbf_w).sum(-1, keepdim=True)        # (N, 1)

    def forward(self, xy):
        alpha = torch.sigmoid(self.alpha_raw)
        return alpha * self.mlp(xy) + (1 - alpha) * self._rbf(xy)


def u_exact(x, y):
    return torch.sin(2 * math.pi * x) * torch.cos(3 * math.pi * y)


def f_rhs(x, y):
    return -13 * math.pi**2 * torch.sin(2 * math.pi * x) * torch.cos(3 * math.pi * y)


def laplacian(model, xy):
    u = model(xy)
    du = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du[:, 0].sum(), xy, create_graph=True)[0][:, 0]
    d2u_dy2 = torch.autograd.grad(du[:, 1].sum(), xy, create_graph=True)[0][:, 1]
    return d2u_dx2 + d2u_dy2


def pde_loss(model, xy_col):
    lap = laplacian(model, xy_col)
    f = f_rhs(xy_col[:, 0], xy_col[:, 1])
    return ((lap - f) ** 2).mean()


def bc_loss(model, xy_bc):
    u_pred = model(xy_bc).squeeze(1)
    u_true = u_exact(xy_bc[:, 0], xy_bc[:, 1])
    return ((u_pred - u_true) ** 2).mean()

