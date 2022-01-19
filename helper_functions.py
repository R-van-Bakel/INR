import torch


def get_coordinates(dim_sizes):
    return torch.cartesian_prod(*map(lambda x: torch.arange(x, dtype=torch.float), dim_sizes)).reshape(*dim_sizes,
                                                                                                       len(dim_sizes))
