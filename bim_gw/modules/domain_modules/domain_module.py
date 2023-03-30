import dataclasses
from typing import Any, Dict, List, Tuple

import torch
from pytorch_lightning import LightningModule

from bim_gw.modules.workspace_encoders import DomainDecoder, DomainEncoder


@dataclasses.dataclass
class DomainSpecs:
    z_size: int
    output_dims: Dict[str, int]
    losses: Dict[str, Any]
    decoder_activation_fn: Dict[str, Any]

    input_keys: List[str]
    latent_keys: List[str]

    requires_acc_computation: bool = False
    workspace_encoder_cls: Any = dataclasses.field(default=DomainEncoder)
    workspace_decoder_cls: Any = dataclasses.field(default=DomainDecoder)


class DomainModule(LightningModule):
    def __init__(self, domain_specs, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.domain_specs = domain_specs

    def forward(self, x: torch.Tensor) -> Tuple:
        return self.encode(x)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def adapt(self, z):
        return z

    def log_domain(self, logger, x, title, max_examples=None, step=None):
        raise NotImplementedError

    def log_domain_from_latent(
        self, logger, z, title, max_examples=None, step=None
    ):
        return self.log_domain(
            logger, self.decode(z), title, max_examples, step=step
        )

    def compute_acc(self, acc_metric, predictions, targets):
        pass

    def loss(self, predictions, targets):
        loss = 0.
        losses = dict()
        for k, loss_fn in enumerate(self.domain_specs.losses):
            loss_val = loss_fn(predictions[k], targets[k]).mean()
            loss += loss_val
            losses[k] = loss_val
        return loss, losses


class PassThroughWM(DomainModule):

    def __init__(self, workspace_module):
        super(PassThroughWM, self).__init__(
            domain_specs=workspace_module.domain_specs
        )
        if (
                len(workspace_module.domain_specs.input_keys)
                != len(workspace_module.domain_specs.latent_keys)
        ):
            raise ValueError(
                "Cannot use PassThroughWM with a domain that has different "
                "number of input and latent keys."
            )
        self.workspace_module = workspace_module
        self.use_pass_through = True

    def pass_through(self, mode=True):
        self.use_pass_through = mode

    def encode(self, x):
        if self.use_pass_through:
            return {
                latent_key: x[input_key]
                for latent_key, input_key in zip(
                    self.workspace_module.domain_specs.latent_keys,
                    self.workspace_module.domain_specs.input_keys
                )
            }
        return self.workspace_module.encode(x)

    def decode(self, z):
        if self.use_pass_through:
            return {
                input_key: z[latent_key]
                for latent_key, input_key in zip(
                    self.workspace_module.domain_specs.latent_keys,
                    self.workspace_module.domain_specs.input_keys
                )
            }
        return self.workspace_module.decode(z)

    def adapt(self, z):
        if self.use_pass_through:
            return z
        return self.workspace_module.adapt(z)

    def compute_acc(self, acc_metric, predictions, targets):
        return self.workspace_module.compute_acc(
            acc_metric, predictions, targets
        )

    def log_domain_from_latent(
        self, logger, z, title, max_examples=None, step=None
    ):
        return self.workspace_module.log_domain_from_latent(
            logger, z, title, max_examples, step=step
        )

    def log_domain(self, logger, x, title, max_examples=None, step=None):
        return self.workspace_module.log_domain(
            logger, x, title, max_examples, step=step
        )
