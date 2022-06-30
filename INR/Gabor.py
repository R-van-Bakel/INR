import sys

if not "./flexconv" in sys.path:
    sys.path.insert(0, './flexconv')

import torch
import wandb

from .utils.gabor_antialiasing import regularize_gabornet
from .INRBaseClass import INRBaseClass
from .GaborModel import GaborModel
from .utils import baseline, cifar_grid


class Gabor(INRBaseClass):
    def __init__(self, domain, codomain, batch_size=None, size=None, **model_kwargs):
        super().__init__(domain, codomain, size)
        self.batch_size = batch_size
        model = GaborModel(dim_linear=self.domain_dim, out_channels=torch.prod(torch.LongTensor(codomain)).item(),
                           batch_size=batch_size, **model_kwargs)
        self.model = model

    def forward(self, coordinates=None):
        if coordinates is None:
            coordinates = self.get_train_coordinates()
        y = super().forward(coordinates)
        return y

    def fit(self, image, optimizer, criterion, scheduler, epochs, image_grid=None, regularize=False,
            log=False, **regularize_kwargs):
        upsampled_image = None
        upsampled_image_grid = None
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
                if log:
                    if upsampled_image is None:
                        upsampled_image = baseline(image, 8).to(image.get_device())
                    if upsampled_image_grid is None:
                        upsampled_image_grid = cifar_grid(256, self.batch_size).to(image_grid.get_device())
                    loss_upsampled = criterion(self(upsampled_image_grid), upsampled_image)
                    wandb.log({"MSE": loss, "MSE (Upsampled)": loss_upsampled})
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            scheduler.step(loss)
        return losses
