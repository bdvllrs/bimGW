from typing import Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule


class WorkspaceModule(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.requires_acc_computation = False

    def forward(self, x: torch.Tensor) -> Tuple:
        return self.encode(x)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def log_domain(self, logger, x, title, max_examples=None):
        raise NotImplementedError

    def compute_acc(self, acc_metric, predictions, targets):
        raise NotImplementedError


class PassThroughWM(WorkspaceModule):
    def __init__(self, workspace_module):
        super(PassThroughWM, self).__init__()
        self.workspace_module = workspace_module
        self.z_size = self.workspace_module.z_size
        self.output_dims = self.workspace_module.output_dims
        self.decoder_activation_fn = self.workspace_module.decoder_activation_fn
        self.losses = self.workspace_module.losses

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

    def log_domain(self, logger, x, title, max_examples=None):
        return self.workspace_module.log_domain(logger, x, title, max_examples)
