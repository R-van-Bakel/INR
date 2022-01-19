import torch
from torch import nn


class INRBaseClass(nn.Module):
    def __init__(self, train_coordinates, codomain_dims, model, **kwargs):
        super().__init__()
        self.model = model

        coordinate_kwargs = {}
        for key, value in kwargs.items():
            if key.startswith("coordinates_"):
                coordinate_kwargs[key[len("coordinates_"):]] = value

        self.domain_dims = train_coordinates.size()
        self.codomain_dims = torch.LongTensor(codomain_dims)
        self.train_coordinates = train_coordinates

    def forward(self, coordinates=None):
        if coordinates is None:
            coordinates = self.train_coordinates
        y = self.model(coordinates)
        y = y.reshape(*coordinates.size()[:-1], *self.codomain_dims)
        return y

    def fit(self, image, optimizer, criterion, scheduler, epoch_size, no_epochs):
        self.train()
        for i in range(no_epochs):
            for j in range(epoch_size):
                optimizer.zero_grad()
                out = self(self.train_coordinates)
                loss = criterion(out, image)
                loss.backward()
                optimizer.step()
            scheduler.step()
