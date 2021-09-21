from typing import Tuple

import torch
from pytorch_lightning import LightningModule


class WorkspaceModule(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.requires_acc_computation = False

    def forward(self, x: torch.Tensor) -> Tuple:
        raise NotImplementedError

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def log_domain(self, logger, x, title, max_examples=None):
        raise NotImplementedError

    def compute_acc(self, acc_metric, predictions, targets):
        raise NotImplementedError
