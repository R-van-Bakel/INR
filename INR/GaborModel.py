import torch
from torch import nn
import numpy as np
from .utils import EinsumLinear


# A slightly altered version of the Gabor model in https://github.com/rjbruin/flexconv/blob/master/ckconv/nn/kernelnet.py


class MFNBase(nn.Module):
    """
    Multiplicative filter network base class.

    Expects the child class to define the 'filters' attribute, which should be
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """

    def __init__(
            self,
            hidden_channels: int,
            out_channels: int,
            no_layers: int,
            weight_scale: float,
            bias: bool,
            batch_size: int = None,
    ):
        super().__init__()

        self.linear = nn.ModuleList(
            [
                EinsumLinear(
                    hidden_channels,
                    hidden_channels,
                    batch_size=batch_size,
                    bias=bias
                )
                for _ in range(no_layers)
            ]
        )
        self.output_linear = EinsumLinear(
            hidden_channels, out_channels, batch_size=batch_size
        )

        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / hidden_channels),
                np.sqrt(weight_scale / hidden_channels),
            )

        return

    def forward(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            lin = self.linear[i - 1](out)
            out = self.filters[i](x) * lin
        out = self.output_linear(out)

        return out


class GaborModel(MFNBase):
    def __init__(
            self,
            dim_linear: int,
            hidden_channels: int,
            out_channels: int,
            no_layers: int,
            covariance: str = "anisotropic",
            input_scale: float = 256.0,
            weight_scale: float = 1.0,
            alpha: float = 6.0,
            beta: float = 1.0,
            bias: bool = True,
            init_spatial_value: float = 1.0,
            final_non_linearity: str = "identity",
            batch_size: int = None,
    ):
        super().__init__(
            hidden_channels,
            out_channels,
            no_layers,
            weight_scale,
            bias,
            batch_size,
        )
        self.filters = nn.ModuleList(
            [
                GaborLayer(
                    dim_linear,
                    hidden_channels,
                    input_scale / np.sqrt(no_layers + 1),
                    alpha / (layer + 1),
                    beta,
                    init_spatial_value,
                    covariance,
                    batch_size,
                )
                for layer in range(no_layers + 1)
            ]
        )
        if final_non_linearity == "identity":
            self.non_linearity = nn.Identity()
        elif final_non_linearity == "sigmoid":
            self.non_linearity = nn.Sigmoid()

    def forward(self, x):
        x = super().forward(x)
        x = self.non_linearity(x)
        return x



class GaborLayer(nn.Module):
    """
    Gabor-like filter as used in GaborNet.
    """

    def __init__(
            self,
            dim_linear: int,
            hidden_channels: int,
            # steerable: bool,
            input_scale: float,
            alpha: float,
            beta: float,
            init_spatial_value: float,
            covariance: bool,
            batch_size: int = 1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.linear = EinsumLinear(dim_linear, hidden_channels, batch_size=batch_size)
        mu = init_spatial_value * (2 * torch.rand(batch_size, hidden_channels, dim_linear) - 1)
        self.mu = nn.Parameter(mu)
        if covariance == "isotropic":
            self.gamma = nn.Parameter(
                torch.distributions.gamma.Gamma(alpha, beta).sample(
                    (batch_size, hidden_channels, 1)
                )
            )
        elif covariance == "anisotropic":
            self.gamma = nn.Parameter(
                torch.distributions.gamma.Gamma(alpha, beta).sample(
                    (batch_size, hidden_channels, dim_linear)
                )
            )
        self.input_scale = input_scale
        self.linear.weight.data *= input_scale * self.gamma
        self.linear.bias.data.uniform_(-np.pi, np.pi)

        # # If steerable, create thetas
        # self.steerable = steerable
        # if self.steerable:
        #     self.theta = nn.Parameter(
        #         torch.rand(
        #             hidden_channels,
        #         )
        #     )

        return

    def forward(self, x):
        n_domain_dims = len(x.size()) - 2
        domain_dims = [1] * n_domain_dims
        # if self.steerable:
        #     gauss_window = rotated_gaussian_window(
        #         x,
        #         self.gamma.view(*domain_dims, *self.gamma.shape),
        #         self.theta,
        #         self.mu.view(*domain_dims, *self.mu.shape),
        #     )
        # else:
        gauss_window = gaussian_window(
            x,
            self.gamma.view(self.batch_size, *domain_dims, *self.gamma.shape[1:]),
            self.mu.view(self.batch_size, *domain_dims, *self.mu.shape[1:])
        )
        return gauss_window * torch.sin(self.linear(x))


def gaussian_window(x, gamma, mu):
    n_domain_dims = len(x.size()) - 2
    return torch.exp(-0.5 * ((gamma * (x.unsqueeze(n_domain_dims + 1) - mu)) ** 2).sum(n_domain_dims + 2))


# def rotation_matrix(theta):
#     cos = torch.cos(theta)
#     sin = torch.sin(theta)
#     return torch.stack([cos, sin, -sin, cos], dim=-1).view(-1, 2, 2)
#
#
# def rotate(theta, input):
#     # theta.shape = [Out, 1]
#     # input.shape = [*B, Channels, 2]
#     # Works for 2-dimensional data
#     possible_domain_dims = "abdefghjklmnpqrstuvwz"
#     domain_dims = possible_domain_dims[:len(input.size()) - 2]
#     return torch.einsum(f"coi, {domain_dims}ci -> {domain_dims}co", rotation_matrix(theta), input)
#
#
# def rotated_gaussian_window(x, gamma, theta, mu):
#     n_domain_dims = len(x.size()) - 1
#     return torch.exp(
#         -0.5 * ((gamma * rotate(2 * np.pi * theta, x.unsqueeze(n_domain_dims) - mu)) ** 2).sum(n_domain_dims + 1)
#     )
