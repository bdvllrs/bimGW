from typing import Tuple

import torch
from pytorch_lightning import LightningModule


class WorkspaceModule(LightningModule):
    def forward(self, x: torch.Tensor) -> Tuple:
        raise NotImplementedError

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

