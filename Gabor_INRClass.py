import sys

if not "./flexconv" in sys.path:
    sys.path.insert(0, './flexconv')

import torch
from torch import nn

from gabor_antialiasing import regularize_gabornet
from INRBaseClass import INRBaseClass
from GaborClass import GaborClass


class Gabor_INRClass(INRBaseClass):
    def __init__(self, train_coordinates, codomain_dims, hidden_channels, no_layers, input_scale,
                 weight_scale, alpha, beta, bias, init_spatial_value, covariance="anisotropic",
                 final_non_linearity="identity", res=32, factor=1.00, target="gabor", method="summed",
                 gauss_stddevs=2.0, gauss_factor=0.01, **kwargs):
        model = GaborClass(dim_linear=train_coordinates.size()[-1], hidden_channels=hidden_channels,
                           out_channels=torch.prod(torch.LongTensor(codomain_dims)).item(), no_layers=no_layers,
                           covariance=covariance, input_scale=input_scale, weight_scale=weight_scale, alpha=alpha,
                           beta=beta, bias=bias, init_spatial_value=init_spatial_value)
        super().__init__(train_coordinates, codomain_dims, model, **kwargs)
        self.res = res
        self.factor = factor
        self.target = target
        self.method = method
        self.gauss_stddevs = gauss_stddevs
        self.gauss_factor = gauss_factor
        if final_non_linearity == "identity":
            self.non_linearity = nn.Identity()
        elif final_non_linearity == "sigmoid":
            self.non_linearity = nn.Sigmoid()

    def forward(self, coordinates=None):
        if coordinates is None:
            coordinates = self.train_coordinates
        y = super().forward(coordinates)
        y = self.non_linearity(y)
        return y

    def fit(self, image, optimizer, criterion, scheduler, epochs, regularize=True):
        self.train()
        losses = []
        for epoch in epochs:
            for j in range(epoch):
                optimizer.zero_grad()
                out = self()
                loss = criterion(out, image)
                if regularize:
                    loss += regularize_gabornet(gabor_net=self.model,
                                                kernel_size=self.res,
                                                factor=self.factor,
                                                target=self.target,
                                                method=self.method,
                                                gauss_stddevs=self.gauss_stddevs,
                                                gauss_factor=self.gauss_factor)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            scheduler.step(loss)
        return losses
