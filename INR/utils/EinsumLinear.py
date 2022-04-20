import torch
from torch import nn
import math


class EinsumLinear(nn.Module):
    def __init__(self, in_features, out_features, batch_size=1, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.weight = nn.parameter.Parameter(torch.empty((batch_size, out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.parameter.Parameter(torch.empty((batch_size, out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Same initialization as torch.nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0,:,:])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, equation=None):
        # If no equation is provided assume shape (*coordinates, dimensionality, n_images),
        # which is used for fitting INRs to images
        if self.bias is None:
            bias = 0
        else:
            bias = self.bias.reshape(self.bias.size(0), *[1]*(len(x.size())-2), self.bias.size(1))
        if equation is None:
            if x.size(0) != self.batch_size:
                raise ValueError(f"The final dimension of x ({x.size(-1)}) should match self.batch_size ({self.batch_size}).")
            return torch.einsum("c...a,c...ba->c...b", x, self.weight) + bias
        else:
            return torch.einsum(equation, x, self.weight) + bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, batch_size={}, bias={}'.format(
            self.in_features, self.out_features, self.batch_size, self.bias is not None
        )
