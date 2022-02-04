import torch
from torch import nn


class INRBaseClass(nn.Module):
    def __init__(self, codomain, train_coordinates=None):
        super().__init__()

        self.codomain = torch.LongTensor(codomain)
        self.train_coordinates = train_coordinates

    def set_train_coordinates(self, train_coordinates):
        self.train_coordinates = train_coordinates

    def get_train_coordinates(self):
        if self.train_coordinates is None:
            raise TypeError("self.train_coordinates is still set to None. Please use self.set_train_coordinates to"
                            "provide proper training coordinates. If calling self.train() training coordinates could be"
                            "passed directly to this method instead.")
        else:
            return self.train_coordinates

    def forward(self, coordinates=None):
        if coordinates is None:
            coordinates = self.get_train_coordinates()
        y = self.model(coordinates)
        y = y.reshape(*coordinates.size()[:-1], *self.codomain)
        return y

    def fit(self, image, optimizer, criterion, scheduler, epoch_size, no_epochs, train_coordinates=None):
        if train_coordinates is None:
            train_coordinates = self.get_train_coordinates()
        self.train()
        for i in range(no_epochs):
            for j in range(epoch_size):
                optimizer.zero_grad()
                out = self(train_coordinates)
                loss = criterion(out, image)
                loss.backward()
                optimizer.step()
            scheduler.step()
