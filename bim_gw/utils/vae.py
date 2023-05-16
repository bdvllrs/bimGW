import numpy as np
import torch
from torch.nn import functional as F


def reparameterize(mean, logvar):
    std = logvar.mul(0.5).exp()
    eps = torch.randn_like(std)
    return eps.mul(std).add(mean)


def gaussian_nll(mu, log_sigma, x):
    # D = mu.size(0) * mu.size(1) * mu.size(2) * mu.size(3)
    r = (
        0.5 * torch.pow((x - mu) / log_sigma.exp(), 2)
        + log_sigma
        + 0.5 * np.log(2 * np.pi)
    )
    # r = D * log_sigma
    return r


def softclip(tensor, min):
    """Clips the tensor values at the minimum value min in a softway. Taken
    from Handful of Trials"""
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor
