from .INRBaseClass import INRBaseClass
from .RFFModel import RFFNet
import wandb
from .utils import baseline, cifar_grid

class RFF(INRBaseClass):
    def __init__(self, domain, codomain, batch_size=None, size=None, **model_kwargs):
        super().__init__(domain, codomain, size)
        self.batch_size = batch_size
        model = RFFNet(self.domain_dim, self.codomain.prod(), batch_size=batch_size, **model_kwargs)
        self.model = model

    def fit(self, image, optimizer, criterion, scheduler, epochs, image_grid=None, log=False):
        upsampled_image = None
        upsampled_image_grid = None
        if image_grid is None:
            image_grid = self.get_train_coordinates()
        self.train()
        losses = []
        for epoch in epochs:
            for i in range(epoch):
                optimizer.zero_grad()
                out = self(image_grid)
                loss = criterion(out, image)
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
