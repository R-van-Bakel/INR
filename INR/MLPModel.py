import torch
from torch import nn
import numpy as np
from .utils import EinsumLinear


class MLPModel(nn.Module):
    def __init__(self, domain, codomain, hidden_size, batch_size=None, activ_func="relu",
                 final_non_linearity="identity", omega0=30.0, omega0_initial=1.0, sigma=1.0):
        super().__init__()
        self.omega0 = omega0
        self.omega0_initial = omega0_initial
        self.sigma = sigma
        if not hidden_size:
            self.lin_layers = nn.ModuleList([EinsumLinear(domain, codomain, batch_size)])
        else:
            self.lin_layers = nn.ModuleList(
                [EinsumLinear(domain, hidden_size[0], batch_size)] +
                [
                    EinsumLinear(hidden_size[i], hidden_size[i+1], batch_size)
                    for i in range(len(hidden_size)-1)
                ]
                + [EinsumLinear(hidden_size[-1], codomain, batch_size)]
            )

        if activ_func == "relu":
            self.non_linearty = torch.nn.ReLU()
        elif activ_func == "sine":
            self.non_linearty = torch.sin
            with torch.no_grad():
                nn.init.uniform_(self.lin_layers[0].weight, -(1 / domain),
                                 (1 / domain))
                nn.init.uniform_(self.lin_layers[0].bias, -(1 / domain),
                                 (1 / domain))
                for i in range(len(self.lin_layers[1:])):
                    layer = self.lin_layers[i+1]
                    hidden_features = hidden_size[i]
                    nn.init.uniform_(layer.weight, -np.sqrt(6 / hidden_features) / omega0,
                                     np.sqrt(6 / hidden_features) / omega0)
                    nn.init.uniform_(layer.bias, -np.sqrt(6 / hidden_features) / omega0, np.sqrt(6 / hidden_features) / omega0)
        elif activ_func == "gaussian":
            self.non_linearty = lambda x: torch.exp(-torch.square(x)/(2*np.square(self.sigma)))

        if final_non_linearity == "identity":
            self.final_non_linearity = nn.Identity()
        elif final_non_linearity == "sigmoid":
            self.final_non_linearity = nn.Sigmoid()

    def forward(self, x):
        if self.omega0_initial is not None:
            x = self.omega0_initial * x
        x = self.lin_layers[0](x)
        if len(self.lin_layers) > 1:
            for layer in self.lin_layers[1:]:
                x = self.non_linearty(x)
                if self.omega0 is not None:
                    x = self.omega0 * x
                x = layer(x)
        x = self.final_non_linearity(x)
        return x
