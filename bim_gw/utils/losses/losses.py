import torch
import torch.distributions as D
from torch.nn import functional as F


def gaussian_ll(mean, logscale, sample):
    logscale = logscale.expand_as(mean)
    dist = D.Normal(mean, torch.exp(logscale))
    logp = dist.log_prob(sample)
    return logp.sum(dim=(1, 2, 3))


def grad_norm(output, parameters, *params, **kwargs):
    gradients = torch.autograd.grad(output, parameters, *params, **kwargs)
    return sum([torch.norm(grad) for grad in gradients if grad is not None])


def cross_entropy(x, y):
    y = torch.argmax(y, 1)
    return F.cross_entropy(x, y)


def nll_loss(x, y):
    y = torch.argmax(y, 1)
    return F.nll_loss(x, y)


loss_functions = {
    "cosine": lambda x, y: -F.cosine_similarity(x, y),
    "mse": F.mse_loss,
    "cross_entropy": cross_entropy,
    "nll": nll_loss
}
