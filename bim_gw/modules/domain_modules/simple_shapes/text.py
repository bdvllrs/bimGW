from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F

from bim_gw.modules.domain_modules.domain_module import (
    DomainModule,
    DomainSpecs,
)
from bim_gw.utils.shapes import generate_dataset
from bim_gw.utils.text_composer.composer import composer
from bim_gw.utils.text_composer.utils import (
    get_choices_from_structure_category,
    inspect_all_choices,
)
from bim_gw.utils.utils import log_if_save_last_images, log_if_save_last_tables
from bim_gw.utils.vae import reparameterize

from ...domain_buffer import DictBuffer
from .attributes import SimpleShapesAttributes


def make_causal_mask_prog(input_dec, encod_out):
    mask = (
        torch.triu(torch.ones(input_dec.size(1), encod_out.size(1))) == 1
    ).permute(1, 0)
    return (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
        .to(input_dec.device)
    )


def convert_angle(angle):
    return angle + 2 * np.pi * (angle < 0)


def symlog(x, alpha=1):
    return (
        torch.sign(x) * torch.log(1 + alpha * torch.abs(x)) / np.log(1 + alpha)
    )


def symexp(x, alpha=1):
    return torch.sign(x) * (torch.exp(alpha * torch.abs(x)) - 1) / alpha


class SymLog(nn.Module):
    def __init__(self, alpha=1):
        super(SymLog, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return symlog(x, self.alpha)

    def inverse(self, x):
        return symexp(x, self.alpha)


class SimpleShapesText(DomainModule):
    def __init__(
        self,
        z_size: int,
        hidden_size: int,
        beta: float,
        n_classes: int,
        imsize: int,
        optim_lr: float = 3e-4,
        optim_weight_decay: float = 1e-5,
        scheduler_step: int = 20,
        scheduler_gamma: float = 0.5,
        train_vae: bool = True,
        train_attr_decoders: bool = True,
        optimize_vae_with_attr_regression: bool = False,
        coef_attr_loss: float = 1,
        coef_vae_loss: float = 1,
    ):
        super(SimpleShapesText, self).__init__(
            DomainSpecs(
                output_dims={"z": z_size},
                decoder_activation_fn={"z": SymLog()},
                losses={"z": F.mse_loss},
                input_keys=["bert", "text", "choices"],
                latent_keys=["z"],
            )
        )

        self.save_hyperparameters(ignore=["domain_examples"])
        self.n_classes = n_classes
        self.bert_size = 768
        self.z_size = z_size
        self.hidden_size = hidden_size
        self.imsize = imsize
        self.train_vae = train_vae
        self.train_attr_decoders = train_attr_decoders
        self.optimize_vae_with_attr_regression = (
            optimize_vae_with_attr_regression
        )
        self.coef_attr_loss = coef_attr_loss
        self.coef_vae_loss = coef_vae_loss

        self.text_composer = composer
        self.composer_inspection = inspect_all_choices(composer)

        self.transformer = None
        self.tokenizer = None

        self.attribute_domain = SimpleShapesAttributes(imsize)
        self.attribute_domain.freeze()

        self.encoder = nn.Sequential(
            nn.Linear(self.bert_size, self.bert_size),
            nn.ReLU(),
            nn.Linear(self.bert_size, self.bert_size // 2),
            nn.ReLU(),
            nn.Linear(self.bert_size // 2, self.z_size * 2),
            SymLog(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_size, self.bert_size // 2),
            nn.ReLU(),
            nn.Linear(self.bert_size // 2, self.bert_size),
            nn.ReLU(),
            nn.Linear(self.bert_size, self.bert_size),
        )

        if not self.train_vae:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            self.decoder.eval()

        self.attribute_encoder = nn.Sequential(
            nn.Linear(self.z_size, self.z_size),
            nn.ReLU(),
            nn.Linear(
                self.z_size,
                sum(self.attribute_domain.domain_specs.output_dims.values()),
            ),
        )
        self.grammar_classifiers = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.z_size, self.z_size),
                    nn.ReLU(),
                    nn.Linear(self.z_size, n_outputs),
                )
                for name, n_outputs in self.composer_inspection.items()
            }
        )

        if not self.train_attr_decoders:
            for param in self.attribute_encoder.parameters():
                param.requires_grad = False
            for param in self.grammar_classifiers.parameters():
                param.requires_grad = False
            self.attribute_encoder.eval()
            self.grammar_classifiers.eval()

        self.grammar_train_acc = nn.ModuleDict(
            {
                name: torchmetrics.Accuracy()
                for name in self.composer_inspection.keys()
            }
        )
        self.grammar_val_acc = nn.ModuleDict(
            {
                name: torchmetrics.Accuracy()
                for name in self.composer_inspection.keys()
            }
        )
        self.grammar_metrics: Dict[str, torchmetrics.Accuracy] = {}

        self.domain_examples: Optional[DictBuffer] = None

        self.register_buffer("log_sigma", torch.tensor(0.0))
        self.register_buffer("beta", torch.tensor(beta))

    def encode(self, text_item: Dict[str, Any]) -> Dict[str, Any]:
        z, _ = self.encode_stats(text_item["bert"])
        return {"z": z}

    def get_sentence_predictions(
        self, z: torch.Tensor, predictions: Dict[str, Any]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        grammar_prediction = self.get_grammar_prediction(z)
        choices = get_choices_from_structure_category(
            self.text_composer, grammar_prediction
        )
        cls = predictions["cls"].detach().cpu().numpy()
        attributes = predictions["attr"].detach().cpu().numpy()
        # Text
        rotation_x = attributes[:, 3] * 2 - 1
        rotation_y = attributes[:, 4] * 2 - 1
        rotations = convert_angle(np.arctan2(rotation_y, rotation_x))

        sentence_predictions, final_choices = [], []
        for k in range(len(cls)):
            sentence, choice = self.text_composer(
                {
                    "shape": int(cls[k]),
                    "rotation": rotations[k],
                    "color": (
                        attributes[k, 5] * 255,
                        attributes[k, 6] * 255,
                        attributes[k, 7] * 255,
                    ),
                    "size": attributes[k, 2],
                    "location": (attributes[k, 0], attributes[k, 1]),
                },
                choices[k],
            )
            sentence_predictions.append(sentence)
            final_choices.append(choice)
        return sentence_predictions, final_choices

    def decode(self, domain_item: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        z_mean = domain_item["z"]
        text_latent = self.decoder(z_mean)
        predictions = self.classify(z_mean)
        predictions = self.attribute_domain.decode(predictions)

        sentence_predictions, final_choices = self.get_sentence_predictions(
            z_mean, predictions
        )
        return {
            "bert": text_latent,
            "text": sentence_predictions,
            "choices": final_choices,
        }

    def sample(
        self,
        size,
        classes=None,
        min_scale=10,
        max_scale=25,
        min_lightness=46,
        max_lightness=256,
    ):
        samples = generate_dataset(
            size,
            min_scale,
            max_scale,
            min_lightness,
            max_lightness,
            32,
            classes,
        )
        cls = samples["classes"]
        x, y = samples["locations"][:, 0], samples["locations"][:, 1]
        size = samples["sizes"]
        rotation = samples["rotations"]
        # assert 0 <= rotation <= 1
        # rotation = rotation * 2 * np.pi / 360  # put in radians
        r = samples["colors"][:, 0]
        g = samples["colors"][:, 1]
        b = samples["colors"][:, 2]

        labels, choices = self.text_composer(
            {
                "shape": cls,
                "rotation": rotation,
                "color": (r, g, b),
                "size": size,
                "location": (x, y),
            }
        )
        return None, labels, choices  # TODO: add BERT vectors

    def metrics(self, mode, predictions, targets):
        if "train" in mode:
            return {}

        if mode not in self.grammar_metrics.keys():
            self.grammar_metrics[mode] = {
                name: torchmetrics.Accuracy().to(predictions["z"].device)
                for name in self.grammar_classifiers.keys()
            }
        grammar_acc = self.grammar_metrics[mode]

        z = predictions["z"]
        target_z = targets["z"]

        metrics = {}

        for grammar_name, classifier in self.grammar_classifiers.items():
            grammar_prediction = classifier(z.detach())
            grammar_target = classifier(target_z.detach()).argmax(-1)
            loss_grammar = F.cross_entropy(grammar_prediction, grammar_target)
            acc = grammar_acc[grammar_name](
                grammar_prediction.softmax(-1), grammar_target
            )

            metrics.update(
                {
                    f"grammar_{grammar_name}_ce": loss_grammar,
                    f"grammar_{grammar_name}_acc": acc,
                }
            )
        return metrics

    def log_domain_from_latent(
        self, logger, z, name, max_examples=None, step=None
    ):
        predictions = self.attribute_domain.decode(self.classify(z["z"]))
        self.attribute_domain.log_domain(
            logger, predictions, name, max_examples, step=step
        )
        sentences, choices = self.get_sentence_predictions(z["z"], predictions)
        text = [[sentence] for sentence in sentences]

        if logger is not None and hasattr(logger, "log_table"):
            logger.log_table(
                name + "_s", columns=["Text"], data=text, step=step
            )

    def log_domain(self, logger, x, name, max_examples=None, step=None):
        self.log_domain_from_latent(
            logger, self.encode(x), name, max_examples, step
        )

    def classify(self, z):
        prediction = self.attribute_encoder(z)
        predictions = {}
        last_dim = 0
        for key in self.attribute_domain.domain_specs.latent_keys:
            dim = self.attribute_domain.domain_specs.output_dims[key]
            act_fn = self.attribute_domain.domain_specs.decoder_activation_fn[
                key
            ]
            pred = act_fn(prediction[:, last_dim : last_dim + dim])
            predictions[key] = pred
            last_dim += dim
        return predictions

    def get_grammar_prediction(self, z):
        return {
            name: torch.argmax(
                self.grammar_classifiers[name](z), dim=1
            ).tolist()
            for name in self.grammar_classifiers.keys()
        }

    def encode_stats(self, text_latent):
        z = self.encoder(text_latent)
        return z[:, : self.z_size], z[:, self.z_size :]

    def reconstruction_loss(
        self, x_reconstructed: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        loss = F.mse_loss(x_reconstructed, x, reduction="sum")
        return loss

    def kl_divergence_loss(
        self, mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return kl

    def step(self, batch, batch_idx, mode="train"):
        domain, targets = batch["t"], batch["attr"]
        bs = domain["bert"].size(0)
        targets = self.attribute_domain.encode(targets.sub_parts)
        x = domain["bert"]
        z_mean, z_logvar = self.encode_stats(x)
        z = reparameterize(z_mean, z_logvar)
        x_reconstructed = self.decoder(z)

        reconstruction_loss = self.reconstruction_loss(x_reconstructed, x)
        kl_divergence_loss = self.kl_divergence_loss(z_mean, z_logvar)
        vae_loss = (reconstruction_loss + self.beta * kl_divergence_loss) / bs

        self.log(
            f"{mode}/reconstruction_loss",
            reconstruction_loss,
            logger=True,
            on_epoch=(mode != "train"),
            batch_size=bs,
        )
        self.log(
            f"{mode}/kl_divergence_loss",
            kl_divergence_loss,
            logger=True,
            on_epoch=(mode != "train"),
            batch_size=bs,
        )
        self.log(
            f"{mode}/vae_loss",
            vae_loss,
            on_epoch=(mode != "train"),
            batch_size=bs,
        )

        z_predictions = z_mean
        if not self.optimize_vae_with_attr_regression:
            z_predictions = z_mean.detach()

        (
            predictions,
            attribute_losses,
            attribute_prediction_loss,
        ) = self.train_attribute_predictions(
            z_predictions, domain, targets, mode=mode
        )
        total_loss = (
            self.coef_vae_loss * vae_loss
            + self.coef_attr_loss * attribute_prediction_loss
        )

        self.log(
            f"{mode}/total_loss",
            total_loss,
            on_epoch=(mode != "train"),
            batch_size=bs,
        )
        return total_loss

    def train_attribute_predictions(self, x, domain, targets, mode="train"):
        bs = domain["bert"].size(0)
        predictions = self.classify(x)
        losses = {}
        total_loss = 0
        for k, (key, pred) in enumerate(predictions.items()):
            loss = self.attribute_domain.domain_specs.losses[key]
            group_loss = loss(pred, targets[key])
            losses[key] = group_loss
            total_loss += group_loss

            self.log(
                f"{mode}/loss_attributes_{k}",
                group_loss,
                logger=True,
                on_epoch=(mode != "train"),
                batch_size=bs,
            )

        grammar_coef = 1 / len(self.grammar_classifiers)
        for name, classifier in self.grammar_classifiers.items():
            grammar_prediction = classifier(x.detach())
            loss_grammar = F.cross_entropy(
                grammar_prediction, domain["choices"][name]
            )
            total_loss += grammar_coef * loss_grammar
            acc_fn = (
                self.grammar_train_acc[name]
                if mode == "train"
                else self.grammar_val_acc[name]
            )
            res = acc_fn(
                grammar_prediction.softmax(-1), domain["choices"][name]
            )
            self.log(
                f"{mode}/loss_grammar_{name}",
                loss_grammar,
                logger=True,
                on_epoch=(mode != "train"),
                batch_size=bs,
            )
            self.log(
                f"{mode}/grammar_{name}_acc",
                res,
                on_epoch=(mode != "train"),
                batch_size=bs,
            )

        self.log(
            f"{mode}/attribute_grammar_loss",
            total_loss,
            on_epoch=(mode != "train"),
            batch_size=bs,
        )

        return predictions, losses, total_loss

    def training_step(self, batch, batch_idx):
        total_loss = self.step(batch, batch_idx, "train")
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss = self.step(batch, batch_idx, "val")
        return total_loss

    def test_step(self, batch, batch_idx):
        total_loss = self.step(batch, batch_idx, "test")
        return total_loss

    def epoch_end(self, mode="val"):
        if self.domain_examples is not None and mode in self.domain_examples:
            domain_examples = self.domain_examples[mode]["in_dist"]
            for logger in self.loggers:
                encoded_s = self.encode(domain_examples["t"])
                predictions = self.classify(encoded_s["z"])
                predictions = self.attribute_domain.decode(predictions)
                sentences, choices = self.get_sentence_predictions(
                    encoded_s["z"], predictions
                )

                text = [[sentence] for sentence in sentences]

                if hasattr(logger, "log_table"):
                    logger.log_table(
                        f"{mode}/predictions_text", columns=["Text"], data=text
                    )

                # Images
                self.attribute_domain.log_domain(
                    logger, predictions, f"{mode}/predictions_reconstruction"
                )

                if self.current_epoch == 0:
                    with log_if_save_last_images(logger):
                        with log_if_save_last_tables(logger):
                            self.attribute_domain.log_domain(
                                logger,
                                domain_examples["attr"].sub_parts,
                                f"{mode}/target_reconstruction",
                            )
                            if hasattr(logger, "log_table"):
                                logger.log_table(
                                    f"{mode}/target_text",
                                    columns=["Text"],
                                    data=[
                                        [domain_examples["t"]["text"][k]]
                                        for k in range(
                                            len(domain_examples["t"]["text"])
                                        )
                                    ],
                                )

    def validation_epoch_end(self, outputs):
        self.epoch_end("val")

    def test_epoch_end(self, outputs):
        self.epoch_end("test")

    def setup(self, stage: Optional[str] = None) -> None:
        if (
            not hasattr(self.trainer, "datamodule")
            or self.domain_examples is not None
            or self.trainer.datamodule is None  # type: ignore
            or not hasattr(
                self.trainer.datamodule,  # type: ignore
                "domain_examples",
            )
        ):
            return
        if stage in ["fit", "validate", "test"]:
            self.domain_examples = DictBuffer(
                self.trainer.datamodule.domain_examples,  # type: ignore
                persistent=False,
            )

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            params,
            lr=self.hparams.optim_lr,
            weight_decay=self.hparams.optim_weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            self.hparams.scheduler_step,
            self.hparams.scheduler_gamma,
        )
        return [optimizer], [scheduler]
