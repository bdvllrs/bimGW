from typing import Tuple

import torch
import torch.nn.functional as F

from bim_gw.modules.workspace_module import WorkspaceModule


class ActionModule(WorkspaceModule):
    def __init__(self):
        super(ActionModule, self).__init__()
        self.z_size = 10

        self.output_dims = [self.z_size]
        self.decoder_activation_fn = [
            None
        ]

        self.losses = [
            F.mse_loss
        ]

    def forward(self, x: torch.Tensor) -> Tuple:
        pass

    def encode(self, x):
        pass

    def decode(self, z):
        pass

    def log_domain(self, logger, x, title, max_examples=None):
        pass

    def compute_acc(self, acc_metric, predictions, targets):
        pass