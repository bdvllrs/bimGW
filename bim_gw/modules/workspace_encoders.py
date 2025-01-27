import torch
from torch import nn


def get_n_layers(n_layers, hidden_size):
    layers = []
    for k in range(n_layers):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
    return layers


class DomainDecoder(torch.nn.Module):
    def __init__(
        self,
        domain_specs,
        in_dim,
        hidden_size,
        n_layers,
        n_layers_head,
    ):
        super(DomainDecoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_layers_head = n_layers_head

        self.item_keys_order = domain_specs.latent_keys
        out_dims = domain_specs.output_dims
        activation_fn = domain_specs.decoder_activation_fn

        if activation_fn is None:
            activation_fn = {
                key: activation_fn for key in self.item_keys_order
            }

        if len(out_dims) != len(activation_fn):
            raise ValueError(
                "The model is missing some loss_functions or "
                "output_dimensions for the outputs."
            )

        self.out_dims = out_dims
        self.activation_fn = activation_fn

        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_size),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_size),
        )

        self.encoder_head = nn.ModuleDict(
            {
                key: nn.Sequential(
                    *get_n_layers(n_layers_head, self.hidden_size),
                    nn.Linear(self.hidden_size, self.out_dims[key]),
                )
                for key in self.item_keys_order
            }
        )

    def forward(self, x):
        out = self.encoder(x)
        outputs = {}
        for key in self.item_keys_order:
            block = self.encoder_head[key]
            activation_fn = self.activation_fn[key]
            z = block(out)
            if activation_fn is not None:
                z = activation_fn(z)
            outputs[key] = z
        return outputs


class DomainEncoder(nn.Module):
    def __init__(self, domain_specs, hidden_size, out_dim, n_layers):
        super(DomainEncoder, self).__init__()
        self.in_dims = domain_specs.output_dims
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.item_keys_order = domain_specs.latent_keys

        self.encoder = nn.Sequential(
            nn.Linear(sum(self.in_dims.values()), self.hidden_size),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_size),
            nn.Linear(self.hidden_size, self.out_dim),
        )

    def forward(self, x):
        assert len(x) == len(self.in_dims), "Not enough values as input."

        x = torch.cat([x[key] for key in self.item_keys_order], dim=-1)
        out = self.encoder(x)
        return out
