import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import EinsumLinear


class RFFEmb(nn.Module):
    def __init__(self, in_features, out_features, sigma, trainable, batch_size=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        self.trainable = trainable
        self.batch_size = batch_size
        B = torch.randn(batch_size, int(out_features / 2), in_features)
        B *= self.sigma*2*math.pi

        self.correction = math.sqrt(2)

        if trainable:
            B = nn.Parameter(B)
        self.register_buffer("B", B, persistent=True)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, sigma={}, trainable={}'.format(
            self.in_features, self.out_features, self.sigma, self.trainable
        )

    def forward(self, x):
        out = torch.einsum("c...a,c...ba->c...b", x, self.B)
        out = torch.cat((out.sin(), out.cos()), dim=-1)
        out = out*self.correction  # Will affect backprop when trainable
        return out


class RFFNet(nn.Module):
    """ RFF net to parameterise convs """

    def __init__(self, in_features, out_features, hidden_features, sigma=1., activation=nn.ReLU, trainable=False,
                 batch_size=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.activation = activation
        self.sigma = sigma

        assert hidden_features[0] % 2 == 0, "Please use an even number of features for the RFF embedding"

        net = []

        dims = [in_features] + list(hidden_features)

        for i in range(len(dims)-1):
            if i == 0:
                net.append(RFFEmb(dims[i], dims[i+1], sigma=sigma, trainable=trainable, batch_size=batch_size))
            else:
                net.append(EinsumLinear(dims[i], dims[i+1], batch_size))
                net.append(activation())


        net.append(EinsumLinear(dims[-1], out_features, batch_size))

        self.net = nn.Sequential(*net)
        self.init()

    def init(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear) or isinstance(layer, EinsumLinear):
                if layer == self.net[-1]:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
                else:
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        out = self.net(x)
        return out


if __name__ == "__main__":
    # Compute correction factor of sqrt(2) for rff emb, check whether it
    # holds for both sin and cos. Of course, the correction will already be applied
    # so we expect std=1.
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm

    in_features = 2
    out_features = 32
    batch_size = 100

    sigmas = np.linspace(0.001, 100, 1000)
    means_sin = []
    means_cos = []
    stds_sin = []
    stds_cos = []

    for sigma in tqdm(sigmas):
        layer = RFFEmb(in_features, out_features, sigma, False)

        out = layer(torch.rand(batch_size, in_features)*2 - 1)
        out = out.chunk(2, dim=-1)

        means_sin.append(out[0].mean())
        means_cos.append(out[1].mean())
        stds_sin.append(out[0].std())
        stds_cos.append(out[1].std())

    plt.subplot(121)
    plt.plot(sigmas, means_sin, label="sin")
    plt.plot(sigmas, means_cos, label="cos")
    plt.xlabel("Sigma")
    plt.ylabel("Mean")
    plt.subplot(122)
    plt.plot(sigmas, stds_sin, label="sin")
    plt.plot(sigmas, stds_cos, label="cos")
    plt.xlabel("Sigma")
    plt.ylabel("Standard deviation")
    plt.legend()
    plt.show()

    batch_size = 100000
    bins = 128
    in_features = 1
    out_features = 128
    hidden_features = [100, 512, 512]
    net = RFFNet(in_features, out_features, hidden_features, sigma=10.)

    x = torch.randn(batch_size, in_features)
    out = net(x)
    print(net)
    print("mean", out.mean().item(), "std", out.std().item())

    plt.hist(out.detach().flatten().numpy(), density=True, alpha=0.5, bins=bins, label="RFF")
    plt.hist(torch.randn(batch_size, out_features).flatten().numpy(),
             density=True, alpha=0.5, bins=bins, label="Standard Gaussian")
    plt.legend()
    plt.show()
