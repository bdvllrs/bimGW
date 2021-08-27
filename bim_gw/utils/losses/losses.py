import torch
import torch.distributions as D


def gaussian_ll(mean, logscale, sample):
    logscale = logscale.expand_as(mean)
    dist = D.Normal(mean, torch.exp(logscale))
    logp = dist.log_prob(sample)
    return logp.sum(dim=(1, 2, 3))


def grad_norm(output, parameters, *params, **kwargs):
    gradients = torch.autograd.grad(output, parameters, *params, **kwargs)
    return sum([torch.norm(grad) for grad in gradients if grad is not None])


