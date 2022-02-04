import sys

if not "./flexconv" in sys.path:
    sys.path.insert(0, './flexconv')

import torch

from .utils.gabor_antialiasing import regularize_gabornet
from .INRBaseClass import INRBaseClass
from .GaborModel import GaborModel


class Gabor(INRBaseClass):
    def __init__(self, domain, codomain, size=None, **model_kwargs):
        super().__init__(domain, codomain, size)
        model = GaborModel(dim_linear=self.domain_dim, out_channels=torch.prod(torch.LongTensor(codomain)).item(),
                           **model_kwargs)
        self.model = model

    def forward(self, coordinates=None):
        if coordinates is None:
            coordinates = self.get_train_coordinates()
        y = super().forward(coordinates)
        return y

    def fit(self, image, optimizer, criterion, scheduler, epochs, image_grid=None, regularize=False,
            **regularize_kwargs):
        if image_grid is None:
            image_grid = self.get_train_coordinates()
        self.train()
        losses = []
        for epoch in epochs:
            for j in range(epoch):
                optimizer.zero_grad()
                out = self(image_grid)
                loss = criterion(out, image)
                if regularize:
                    loss += regularize_gabornet(gabor_net=self.model,
                                                **regularize_kwargs)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            scheduler.step(loss)
        return losses
