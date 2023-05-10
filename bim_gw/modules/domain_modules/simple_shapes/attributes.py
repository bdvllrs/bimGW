from typing import Dict

import numpy as np
import torch
from torch.nn import functional as F

from bim_gw.modules.domain_modules.domain_module import (
    DomainModule,
    DomainSpecs
)
from bim_gw.utils.losses.info_nce import InfoNCE
from bim_gw.utils.losses.losses import nll_loss
from bim_gw.utils.shapes import generate_dataset, log_shape_fig


class SimpleShapesAttributes(DomainModule):
    def __init__(self, imsize, info_nce_temp=0.1):
        self.n_classes = 3
        self.imsize = imsize
        self.info_nce_temp = info_nce_temp
        z_size = 8

        super(SimpleShapesAttributes, self).__init__(
            DomainSpecs(
                output_dims={
                    "z_cls": self.n_classes, "z_attr": z_size
                },
                decoder_activation_fn={
                    "z_cls": lambda x: torch.log_softmax(x, dim=1),  # shapes
                    "z_attr": torch.tanh,  # rest
                },
                losses={
                    "z_cls": lambda x, y: nll_loss(x, y),  # shapes
                    "z_attr": InfoNCE(self.info_nce_temp),  # rest
                },
                input_keys=["cls", "attr"],
                latent_keys=["z_cls", "z_attr"],
                requires_acc_computation=True
            )
        )
        self.save_hyperparameters("imsize", "info_nce_temp")

    def encode(self, x: Dict[str, torch.Tensor]):
        out_latents = x['attr'].clone()
        out_latents[:, 0] = out_latents[:, 0] / self.imsize
        out_latents[:, 1] = out_latents[:, 1] / self.imsize
        out_latents[:, 2] = out_latents[:, 2] / self.imsize
        out_latents = out_latents * 2 - 1
        return {
            "z_cls": F.one_hot(x['cls'], self.n_classes).type_as(out_latents),
            "z_attr": out_latents,
        }

    def decode(self, x: Dict[str, torch.Tensor]):
        out_latents = x['z_attr'].clone()
        out_latents = (out_latents + 1) / 2
        out_latents[:, 0] = out_latents[:, 0] * self.imsize
        out_latents[:, 1] = out_latents[:, 1] * self.imsize
        out_latents[:, 2] = out_latents[:, 2] * self.imsize
        return {
            "cls": torch.argmax(x['z_cls'], dim=-1),
            "attr": out_latents,
        }

    def adapt(self, x: Dict[str, torch.Tensor]):
        return {
            "z_cls": x['z_cls'].exp(),
            "z_attr": x['z_attr'],
        }

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
        classes = x["cls"][:max_examples].detach().cpu().numpy()
        latents = x["attr"][:max_examples].detach().cpu().numpy()

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

    # def loss(self, predictions, targets):
    #     loss, losses = super().loss(predictions, targets)
    #     # if self.use_unpaired:
    #     #     # Add unpaired loss label
    #     #     # Do not add it to loss, as it is already counted. Only for
    #     #     logging.
    #     #     losses["unpaired"] = F.mse_loss(predictions[1][:, -1],
    #     #     targets[1][:, -1])
    #     return loss, losses
