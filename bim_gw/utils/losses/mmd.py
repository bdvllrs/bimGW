import torch
from torch import nn


def mmd_loss(x, target, bandwidth=None, squared_mmd=False):
    bandwidth = bandwidth or [2, 5, 10, 20, 40, 80]

    bandwidth = bandwidth if type(bandwidth) == list else [bandwidth]
    n = x.size(0)
    m = target.size(0)
    X = torch.cat((x, target), 0)
    XX = X @ X.t()
    X2 = torch.sum(X * X, dim=1, keepdim=True)

    K = XX - 0.5 * X2 - 0.5 * X2.t()

    s1 = torch.ones((n, 1)) / n
    s2 = -torch.ones((m, 1)) / m
    s = torch.cat((s1, s2), 0)
    S = s @ s.t()

    loss = torch.tensor(0.0).to(x.device)

    S = S.to(x.device)

    for i in range(len(bandwidth)):
        k = S * torch.exp(K / bandwidth[i])
        loss += k.sum()

    if squared_mmd:
        return loss
    return torch.sqrt(loss)


class MMDLoss(nn.Module):
    def __init__(self, bandwidth=None, squared_mmd=False):
        """
        MMDLoss with gaussian kernel as defined in
        http://www.gatsby.ucl.ac.uk/~gretton/papers/GreBorRasSchSmo07.pdf
        Args:
            bandwidth: coefficient for the gaussian kernel
            squared_mmd: if true, use the MMD squared loss. Defaults to the
            sqrt(MMDLoss) as explained in
            http://proceedings.mlr.press/v37/li15.pdf
        """
        super(MMDLoss, self).__init__()

        self.squared_mmd = squared_mmd
        self.bandwidth = bandwidth if type(bandwidth) == list else [bandwidth]

    def forward(self, x, target):
        mmd_loss(x, target, self.bandwidth, self.squared_mmd)
