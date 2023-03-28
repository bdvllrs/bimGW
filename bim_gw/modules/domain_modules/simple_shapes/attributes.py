import numpy as np
import torch
from torch.nn import functional as F

from bim_gw.datasets.domain import DomainItems
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
        ]
        self.requires_acc_computation = True
        self.decoder_activation_fn = [
            lambda x: torch.log_softmax(x, dim=1),  # shapes
            torch.tanh,  # rest
        ]

        self.losses = [
            lambda x, y: nll_loss(x, y),  # shapes
            F.mse_loss,  # rest
        ]

    def encode(self, x: DomainItems) -> DomainItems:
        out_latents = x.attributes.clone()
        out_latents[:, 0] = out_latents[:, 0] / self.imsize
        out_latents[:, 1] = out_latents[:, 1] / self.imsize
        out_latents[:, 2] = out_latents[:, 2] / self.imsize
        out_latents = out_latents * 2 - 1
        return DomainItems(
            x.available_masks,
            cls=F.one_hot(x.cls, self.n_classes).type_as(x.attributes),
            attributes=out_latents,
        )

    def decode(self, x: DomainItems) -> DomainItems:
        out_latents = x.attributes.clone()
        out_latents = (out_latents + 1) / 2
        out_latents[:, 0] = out_latents[:, 0] * self.imsize
        out_latents[:, 1] = out_latents[:, 1] * self.imsize
        out_latents[:, 2] = out_latents[:, 2] * self.imsize
        return DomainItems(
            x.available_masks,
            cls=torch.argmax(x.cls, dim=-1),
            attributes=out_latents,
        )

    def adapt(self, x: DomainItems) -> DomainItems:
        return DomainItems(
            x.available_masks,
            cls=x.cls.exp(),
            attributes=x.attributes,
        )

    def compute_acc(self, acc_metric, predictions, targets):
        return acc_metric(predictions[0], targets[0].to(torch.int16))

    def sample(
        self, size, classes=None, min_scale=10, max_scale=25, min_lightness=46,
        max_lightness=256
    ):
        samples = generate_dataset(
            size, min_scale, max_scale, min_lightness, max_lightness, 32,
            classes
        )
        cls = samples["classes"]
        x, y = samples["locations"][:, 0], samples["locations"][:, 1]
        radius = samples["sizes"]
        rotation = samples["rotations"]
        rotation_x = (np.cos(rotation) + 1) / 2
        rotation_y = (np.sin(rotation) + 1) / 2

        r = samples["colors"][:, 0]
        g = samples["colors"][:, 1]
        b = samples["colors"][:, 2]

        labels = (
            torch.from_numpy(cls),
            torch.from_numpy(
                np.stack(
                    [x, y, radius, rotation_x, rotation_y, r, g, b], axis=1
                )
            ).to(torch.float),
        )
        return labels

    def log_domain(self, logger, x, name, max_examples=None, step=None):
        classes = x[0][:max_examples].detach().cpu().numpy()
        latents = x[1][:max_examples].detach().cpu().numpy()

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
        labels = ["c", "x", "y", "s", "rotx", "roty", "r", "g", "b"]
        # if self.use_unpaired:
        #     labels.append("u")
        text = []
        for k in range(len(classes)):
            text.append([classes[k].item()] + latents[k].tolist())
        if logger is not None and hasattr(logger, "log_table"):
            logger.log_table(
                name + "_text", columns=labels, data=text, step=step
            )
        else:
            print(labels)
            print(text)

    def loss(self, predictions, targets):
        loss, losses = super().loss(predictions, targets)
        # if self.use_unpaired:
        #     # Add unpaired loss label
        #     # Do not add it to loss, as it is already counted. Only for
        #     logging.
        #     losses["unpaired"] = F.mse_loss(predictions[1][:, -1],
        #     targets[1][:, -1])
        return loss, losses
