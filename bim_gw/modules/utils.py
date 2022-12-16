import torch
from torch import nn


def get_n_layers(n_layers, hidden_size):
    layers = []
    for k in range(n_layers):
        layers.extend([
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        ])
    return layers


class DomainDecoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_size, n_layers, n_layers_head, out_dims=0, activation_fn=None):
        super(DomainDecoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_layers_head = n_layers_head

        if isinstance(out_dims, int):
            out_dims = [out_dims]
        if not isinstance(activation_fn, (list, tuple)):
            activation_fn = [activation_fn]

        assert len(out_dims) == len(
            activation_fn), "The model is missing some loss_functions or output_dimensions for the outputs."

        self.out_dims = out_dims
        self.activation_fn = activation_fn

        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_size),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_size)
        )

        self.encoder_head = nn.ModuleList([
            nn.Sequential(
                *get_n_layers(n_layers_head, self.hidden_size),
                nn.Linear(self.hidden_size, pose_dim),
            )
            for pose_dim in self.out_dims
        ])

    def forward(self, x):
        out = self.encoder(x)
        outputs = []
        for block, activation_fn in zip(self.encoder_head, self.activation_fn):
            z = block(out)
            if activation_fn is not None:
                z = activation_fn(z)
            outputs.append(z)
        return outputs


class DomainEncoder(nn.Module):
    def __init__(self, in_dims, hidden_size, out_dim, n_layers):
        super(DomainEncoder, self).__init__()
        if isinstance(in_dims, int):
            in_dims = [in_dims]
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = nn.Sequential(
            nn.Linear(sum(self.in_dims), self.hidden_size),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_size),
            nn.Linear(self.hidden_size, self.out_dim)
        )

    def forward(self, x):
        if len(self.in_dims) > 1:
            assert len(x) == len(self.in_dims), "Not enough values as input."
        x = torch.cat(x, dim=-1)
        out = self.encoder(x)
        return out

class LMDomainEncoder(DomainEncoder):
    def __init__(self, in_dims, hidden_size, out_dim, n_layers):
        super(LMDomainEncoder, self).__init__(in_dims, hidden_size, out_dim, n_layers)
        z_size = sum(in_dims)
        self.encoder = nn.Sequential(
            nn.Linear(z_size, z_size),
            nn.ReLU(),
            nn.Linear(z_size, z_size // 2),
            nn.ReLU(),
            nn.Linear(z_size // 2, self.out_dim),
        )

class LMDomainDecoder(DomainDecoder):
    def __init__(self, in_dim, hidden_size, n_layers, n_layers_head, out_dims=0, activation_fn=None):
        super(LMDomainDecoder, self).__init__(in_dim, hidden_size, n_layers, n_layers_head, out_dims, activation_fn)

        self.encoder_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, pose_dim // 2),
                nn.ReLU(),
                nn.Linear(pose_dim // 2, pose_dim),
                nn.ReLU(),
                nn.Linear(pose_dim, pose_dim),
            )
            for pose_dim in self.out_dims
        ])


def mask_predictions(predictions, targets, mask):
    for k in range(len(predictions)):
        predictions[k] = (~mask) * predictions[k] + mask * targets[k]
    return predictions
