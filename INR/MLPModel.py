import torch
from torch import nn
import numpy as np


class MLPModel(nn.Module):
    def __init__(self, domain, codomain, hidden_size, activ_func="relu",
                 final_non_linearity="identity", omega0=30.0, omega0_initial=1.0):
        super().__init__()
        self.omega0 = omega0
        self.omega0_initial = omega0_initial

        if not hidden_size:
            self.lin_layers = nn.ModuleList([torch.nn.Linear(domain, codomain)])
        else:
            self.lin_layers = nn.ModuleList(
                [nn.Linear(domain, hidden_size[0])] +
                [
                    nn.Linear(hidden_features, hidden_features)
                    for hidden_features in hidden_size[1:-2]
                ]
                + [nn.Linear(hidden_size[-1], codomain)]
            )

        if activ_func == "relu":
            self.non_linearty = torch.nn.ReLU()
        elif activ_func == "sine":
            self.non_linearty = torch.sin
            with torch.no_grad():
                nn.init.uniform_(self.lin_layers[0].weight, -np.sqrt(1 / domain) / omega0_initial,
                                 np.sqrt(1 / domain) / omega0_initial)
                nn.init.uniform_(self.lin_layers[0].bias, -np.sqrt(1 / domain) / omega0_initial,
                                 np.sqrt(1 / domain) / omega0_initial)
                for i in range(len(self.lin_layers[1:])):
                    layer = self.lin_layers[i+1]
                    hidden_features = hidden_size[i]
                    print(hidden_features)
                    nn.init.uniform_(layer.weight, -np.sqrt(6 / hidden_features) / omega0,
                                     np.sqrt(6 / hidden_features) / omega0)
                    nn.init.uniform_(layer.bias, -np.sqrt(6 / hidden_features) / omega0, np.sqrt(6 / hidden_features) / omega0)

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
