from typing import Tuple

import torch
from pytorch_lightning import LightningModule

from bim_gw.modules.workspace_encoders import DomainEncoder, DomainDecoder


class DomainModule(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.workspace_encoder_cls = DomainEncoder
        self.workspace_decoder_cls = DomainDecoder
        self.requires_acc_computation = False

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

    def compute_acc(self, acc_metric, predictions, targets):
        raise NotImplementedError

    def loss(self, predictions, targets):
        loss = 0.
        losses = dict()
        for k, loss_fn in enumerate(self.losses):
            l = loss_fn(predictions[k], targets[k]).mean()
            loss += l
            losses[k] = l
        return loss, losses


class PassThroughWM(DomainModule):
    def __init__(self, workspace_module):
        super(PassThroughWM, self).__init__()
        self.workspace_module = workspace_module
        self.z_size = self.workspace_module.z_size
        self.output_dims = self.workspace_module.output_dims
        self.decoder_activation_fn = self.workspace_module.decoder_activation_fn
        self.losses = self.workspace_module.losses
        self.workspace_encoder_cls = self.workspace_module.workspace_encoder_cls
        self.workspace_decoder_cls = self.workspace_module.workspace_decoder_cls
        # self.requires_acc_computation = self.workspace_module.requires_acc_computation

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

    def log_domain(self, logger, x, title, max_examples=None, step=None):
        return self.workspace_module.log_domain(logger, x, title, max_examples, step=step)
