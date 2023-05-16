import dataclasses
from typing import Any, Dict, List, Tuple

import torch
from pytorch_lightning import LightningModule

from bim_gw.modules.workspace_encoders import DomainDecoder, DomainEncoder


@dataclasses.dataclass
class DomainSpecs:
    output_dims: Dict[str, int]
    losses: Dict[str, Any]
    decoder_activation_fn: Dict[str, Any]

    input_keys: List[str]
    latent_keys: List[str]

    requires_acc_computation: bool = False
    workspace_encoder_cls: Any = dataclasses.field(default=DomainEncoder)
    workspace_decoder_cls: Any = dataclasses.field(default=DomainDecoder)


class DomainModule(LightningModule):
    def __init__(self, domain_specs: DomainSpecs, *args, **kwargs):
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
        loss = 0.0
        losses = dict()
        for k, key in enumerate(self.domain_specs.latent_keys):
            loss_fn = self.domain_specs.losses[key]
            loss_val = loss_fn(predictions[key], targets[key]).mean()
            loss += loss_val
            # use k instead of key to keep the same logging keys
            losses[k] = loss_val
        return loss, losses


class PassThroughWM(DomainModule):
    def __init__(self, workspace_module):
        super(PassThroughWM, self).__init__(
            domain_specs=workspace_module.domain_specs
        )
        self.workspace_module = workspace_module
        self.use_pass_through = True

    def pass_through(self, mode=True):
        self.use_pass_through = mode

    def encode(self, x):
        if self.use_pass_through:
            return x
        return self.workspace_module.encode(x)

    def decode(self, z):
        if self.use_pass_through:
            return z
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
