import torch


def coordinate_grid(domain, size=None, reshape=True):
    domain = torch.FloatTensor(domain)
    if size is None:
        size = torch.round((domain.T[1] - domain.T[0]).T).long() + 1
    else:
        if isinstance(size, int):
            size = [size]*domain.size(0)
        size = torch.LongTensor(size).unsqueeze(1)
    zipped_params = zip(domain, size)
    lin_spaces = [torch.linspace(*params[0].tolist(), params[1].item()) for params in zipped_params]
    grid = torch.cartesian_prod(*lin_spaces)
    if reshape:
        grid = grid.reshape((*size, domain.size(0)))
    return grid


def cifar_grid(size):
    return torch.cartesian_prod(*map(lambda x: torch.arange(x, dtype=torch.float), [size, size]))\
               .reshape(size, size, 2)*(32/size)
