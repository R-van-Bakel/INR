from .INRBaseClass import INRBaseClass
from .RFFModel import RFFNet


class RFF(INRBaseClass):
    def __init__(self, domain, codomain, size=None, **model_kwargs):
        super().__init__(domain, codomain, size)
        model = RFFNet(self.domain_dim, self.codomain.prod(), **model_kwargs)
        self.model = model

    def fit(self, image, optimizer, criterion, scheduler, epochs, image_grid=None):
        if image_grid is None:
            image_grid = self.get_train_coordinates()
        self.train()
        losses = []
        for epoch in epochs:
            for i in range(epoch):
                optimizer.zero_grad()
                out = self(image_grid)
                loss = criterion(out, image)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            scheduler.step(loss)
        return losses