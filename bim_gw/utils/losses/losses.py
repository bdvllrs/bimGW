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


def vis_to_text_accuracy(gw, acc_fn, vis_domain, targets):
    # translate the visual domain to text domain
    predicted_t = gw.translate(vis_domain, "v", "t")[0]
    # get the word prediction from the predicted
    logits = gw.domain_mods["t"].decode(predicted_t).softmax(dim=-1)
    return acc_fn(logits, gw.domain_mods["t"].get_targets(targets))
