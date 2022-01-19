import torch
from INRBaseClass import INRBaseClass
from MLPClass import MLPClass


class MLP_INRClass(INRBaseClass):
    def __init__(self, train_coordinates, codomain_dims, hidden_size=15, no_layers=5, activ_func="relu",
                 final_non_linearity="identity", omega0=30.0, omega0_initial=1.0, **kwargs):
        model = MLPClass(len(train_coordinates.size()) - 1, torch.prod(torch.LongTensor(codomain_dims)),
                         hidden_size, no_layers, activ_func, final_non_linearity, omega0, omega0_initial)
        super().__init__(train_coordinates, codomain_dims, model, **kwargs)
        if activ_func == "sine":
            # self.nyquist_freq = MLP_INRClass.calc_nyquist_freq(self.train_coordinates.flatten(0, -2))  # Flatten if non-flattened coordinates are used
            self.use_sine = True
        else:
            self.use_sine = False

    def get_train_coordinates(self, domain_dims, **kwargs):
        return MLP_INRClass.get_coordinates(dims=domain_dims, **kwargs)

    def fit(self, image, optimizer, criterion, scheduler, epochs):  # , regularize=True):
        self.train()
        losses = []
        for epoch in epochs:
            for i in range(epoch):
                optimizer.zero_grad()
                out = self()
                loss = criterion(out, image)
                # if regularize and self.use_sine:
                #     loss += 1 * my_MLP_INRClass.nyquist_loss(self.model.lin_layers[0].weight.T, self.nyquist_freq,
                #                                                 self.model.omega0_initial)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            scheduler.step(loss)
        return losses

    # @staticmethod
    # def calc_nyquist_freq(coordinates):
    #     dim_sizes = []
    #     for dim in coordinates.T:
    #         dim_sizes.append(len(set(dim.tolist())))
    #     dim_sizes = torch.LongTensor(dim_sizes)
    #     dim_diffs = torch.max(coordinates, 0)[0] - torch.min(coordinates, 0)[0] + 1
    #     freqs = (dim_diffs/dim_sizes)/2
    #     return torch.min(freqs).item()

    # @staticmethod
    # def nyquist_loss(weight_matrix, nyquist_freq, omega0=0):
    #     weight_freqs = (2*torch.pi)/(omega0*torch.abs(torch.sum(weight_matrix, 0)))
    #     return torch.mean(torch.nn.functional.relu(weight_freqs - nyquist_freq))
