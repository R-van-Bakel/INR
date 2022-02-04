import torch
from torch import nn
from helper_functions import coordinate_grid


class INRBaseClass(nn.Module):
    def __init__(self, domain, codomain, size=None):
        super().__init__()

        if isinstance(domain, int):
            self.domain = None
            self.domain_dim = domain
        else:
            domain = torch.LongTensor(domain)
            self.domain = domain
            self.domain_dim = domain.size(-1).item()

        self.codomain = torch.LongTensor(codomain)

        if size is not None:
            size = torch.FloatTensor(size)
        self.size = size

    def set_domain(self, domain):
        domain = torch.LongTensor(domain)
        if not domain.size(-1).item() == self.domain_dim:
            raise ValueError("The given domain specification does not comply with the previously given domain_dim.")
        self.domain = domain

    def set_size(self, size):
        size = torch.FloatTensor(size)
        self.size = size

    def get_train_coordinates(self):
        if self.domain is None:
            raise TypeError("self.domain is still set to None. Please use self.set_domain to provide proper training"
                            "coordinates. If calling self.train() training coordinates could be passed directly to this"
                            "method instead.")
        elif self.size is None:
            return coordinate_grid(self.domain)
        else:
            return coordinate_grid(self.domain, self.size)

    def forward(self, coordinates=None):
        if coordinates is None:
            coordinates = self.get_train_coordinates()
        y = self.model(coordinates)
        y = y.reshape(*coordinates.size()[:-1], *self.codomain)
        return y

    def fit(self, image, optimizer, criterion, scheduler, epoch_size, no_epochs, image_grid=None):
        if image_grid is None:
            image_grid = self.get_train_coordinates()
        self.train()
        for i in range(no_epochs):
            for j in range(epoch_size):
                optimizer.zero_grad()
                out = self(image_grid)
                loss = criterion(out, image)
                loss.backward()
                optimizer.step()
            scheduler.step()
