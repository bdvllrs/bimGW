import torch
import torch.nn.functional as F

from bim_gw.modules.workspace_module import WorkspaceModule
from bim_gw.utils.losses.losses import nll_loss


class ActionModule(WorkspaceModule):
    def __init__(self, n_classes, imsize):
        super(ActionModule, self).__init__()
        self.n_classes = n_classes + 1  # +1 for the "no transformation" class
        self.z_size = 8
        self.imsize = imsize

        self.output_dims = [1, self.n_classes, self.z_size]
        self.requires_acc_computation = True
        self.decoder_activation_fn = [
            torch.sigmoid,
            lambda x: torch.softmax(x, dim=1),  # shapes
            # torch.tanh,  # rotations
            torch.tanh,  # rest
        ]

        self.losses = [
            F.binary_cross_entropy,
            lambda x, y: nll_loss(x.log(), y),  # shapes
            # F.mse_loss,  # rotations
            F.mse_loss  # rest
        ]

    def encode(self, x):
        return self(x)

    def decode(self, x):
        is_active, logits, latent = x
        out_latents = latent.clone()
        out_latents[:, 0] = out_latents[:, 0] * self.imsize
        out_latents[:, 1] = out_latents[:, 1] * self.imsize
        out_latents[:, 2] = out_latents[:, 2] * self.imsize
        return is_active, torch.argmax(logits, dim=-1), out_latents

    def forward(self, x: list):
        is_active, cls, latents = x
        out_latents = latents.clone()
        out_latents[:, 0] = out_latents[:, 0] / self.imsize
        out_latents[:, 1] = out_latents[:, 1] / self.imsize
        out_latents[:, 2] = out_latents[:, 2] / self.imsize
        return is_active.reshape(-1, 1).type_as(latents), torch.nn.functional.one_hot(cls, self.n_classes).type_as(latents), out_latents

    def compute_acc(self, acc_metric, predictions, targets):
        return acc_metric(predictions[0], targets[0].to(torch.int16))

    def log_domain(self, logger, x, name, max_examples=None):
        classes = x[1][:max_examples].detach().cpu().numpy()
        latents = x[2][:max_examples].detach().cpu().numpy()

        for k in range(classes.shape[0]):
            text = ", ".join(map(str, [classes[k].item()] + latents[k].tolist()))
            logger.experiment[name + "_text"].log(text)