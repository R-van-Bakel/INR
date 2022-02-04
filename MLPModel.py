import torch
from torch import nn
import numpy as np


class MLPModel(nn.Module):
    def __init__(self, domain_size=2, codomain_size=3, hidden_size=15, no_layers=5, activ_func="relu",
                 final_non_linearity="identity", omega0=30.0, omega0_initial=1.0):
        super().__init__()
        self.omega0 = omega0
        self.omega0_initial = omega0_initial

        if no_layers == 1:
            self.lin_layers = nn.ModuleList([torch.nn.Linear(domain_size, codomain_size)])
        else:
            self.lin_layers = nn.ModuleList(
                [nn.Linear(domain_size, hidden_size)] +
                [
                    nn.Linear(hidden_size, hidden_size)
                    for layer in range(no_layers - 2)
                ]
                + [nn.Linear(hidden_size, codomain_size)]
            )

        if activ_func == "relu":
            self.non_linearty = torch.nn.ReLU()
        elif activ_func == "sine":
            self.non_linearty = torch.sin
            with torch.no_grad():
                nn.init.uniform_(self.lin_layers[0].weight, -np.sqrt(1 / domain_size) / omega0_initial,
                                 np.sqrt(1 / domain_size) / omega0_initial)
                nn.init.uniform_(self.lin_layers[0].bias, -np.sqrt(1 / domain_size) / omega0_initial,
                                 np.sqrt(1 / domain_size) / omega0_initial)
                for layer in self.lin_layers[1:]:
                    nn.init.uniform_(layer.weight, -np.sqrt(6 / hidden_size) / omega0,
                                     np.sqrt(6 / hidden_size) / omega0)
                    nn.init.uniform_(layer.bias, -np.sqrt(6 / hidden_size) / omega0, np.sqrt(6 / hidden_size) / omega0)

        if final_non_linearity == "identity":
            self.final_non_linearity = nn.Identity()
        elif final_non_linearity == "sigmoid":
            self.final_non_linearity = nn.Sigmoid()

    def forward(self, x):
        if self.omega0_initial is not None:
            x = self.omega0_initial * x
        for layer in self.lin_layers:
            x = self.non_linearty(x)
            if self.omega0 is not None:
                x = self.omega0 * x
            x = layer(x)
        x = torch.sigmoid(x)
        return x
