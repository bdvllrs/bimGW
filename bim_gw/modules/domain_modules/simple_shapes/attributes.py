import numpy as np
import torch
from torch.nn import functional as F

from bim_gw.modules.domain_modules.domain_module import DomainModule
from bim_gw.utils.losses.losses import nll_loss
from bim_gw.utils.shapes import generate_dataset, log_shape_fig


class SimpleShapesAttributes(DomainModule):
    def __init__(self, imsize):
        super(SimpleShapesAttributes, self).__init__()
        self.save_hyperparameters()

        self.n_classes = 3
        self.z_size = 8
        self.imsize = imsize

        self.output_dims = [
            self.n_classes,
            self.z_size,
            # 1
        ]
        self.requires_acc_computation = True
        self.decoder_activation_fn = [
            lambda x: torch.log_softmax(x, dim=1),  # shapes
            torch.tanh,  # rest
            # torch.tanh  # unpaired
        ]

        self.losses = [
            lambda x, y: nll_loss(x, y),  # shapes
            F.mse_loss,  # rest
            # F.mse_loss  # unpaired
        ]

    def encode(self, x):
        if len(x) == 2:
            cls, latents = x
            # unpaired = torch.ones_like(latents[:, 0]) * 0.5
        else:
            cls, latents, unpaired = x
        out_latents = latents.clone()
        out_latents[:, 0] = out_latents[:, 0] / self.imsize
        out_latents[:, 1] = out_latents[:, 1] / self.imsize
        out_latents[:, 2] = out_latents[:, 2] / self.imsize
        return (torch.nn.functional.one_hot(cls, self.n_classes).type_as(latents),
                # rotations,
                out_latents * 2 - 1)
                # unpaired * 2 - 1)

    def decode(self, x):
        if len(x) == 2:
            logits, latents = x
            # unpaired = torch.zeros_like(latents[:, 0])
        else:
            logits, latents, unpaired = x
        out_latents = (latents.clone() + 1) / 2
        out_latents[:, 0] = out_latents[:, 0] * self.imsize
        out_latents[:, 1] = out_latents[:, 1] * self.imsize
        out_latents[:, 2] = out_latents[:, 2] * self.imsize
        return (torch.argmax(logits, dim=-1),
                out_latents)
                # (unpaired + 1) / 2)

    def adapt(self, x):
        if len(x) == 2:
            return x[0].exp(), x[1]
        else:
            return x[0].exp(), x[1], x[2]

    def compute_acc(self, acc_metric, predictions, targets):
        return acc_metric(predictions[0], targets[0].to(torch.int16))

    def sample(self, size, classes=None, min_scale=10, max_scale=25, min_lightness=46, max_lightness=256):
        samples = generate_dataset(size, min_scale, max_scale, min_lightness, max_lightness, 32, classes)
        cls = samples["classes"]
        x, y = samples["locations"][:, 0], samples["locations"][:, 1]
        radius = samples["sizes"]
        rotation = samples["rotations"]
        rotation_x = (np.cos(rotation) + 1) / 2
        rotation_y = (np.sin(rotation) + 1) / 2
        # assert 0 <= rotation <= 1
        # rotation = rotation * 2 * np.pi / 360  # put in radians
        r, g, b = samples["colors"][:, 0], samples["colors"][:, 1], samples["colors"][:, 2]

        labels = (
            torch.from_numpy(cls),
            torch.from_numpy(np.stack([x, y, radius, rotation_x, rotation_y, r, g, b], axis=1)).to(torch.float),
        )
        return labels

    def log_domain(self, logger, x, name, max_examples=None, step=None):
        classes = x[0][:max_examples].detach().cpu().numpy()
        latents = x[1][:max_examples].detach().cpu().numpy()
        unpaired = np.zeros_like(latents[:, 0])
        if len(x) == 3:
            unpaired = x[2][:max_examples].detach().cpu().numpy()

        # visualization
        log_shape_fig(
            logger,
            classes,
            # rotations,
            latents,
            name + "_vis",
            step
        )

        # text
        labels = ["c", "x", "y", "s", "rotx", "roty", "r", "g", "b", "u"]
        text = []
        for k in range(len(classes)):
            text.append([classes[k].item()] + latents[k].tolist() + [unpaired[k].item()])
        if logger is not None:
            logger.log_table(name + "_text", columns=labels, data=text, step=step)
        else:
            print(labels)
            print(text)
