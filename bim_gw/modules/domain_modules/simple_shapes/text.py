import numpy as np
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F

from bim_gw.modules.domain_modules.domain_module import DomainModule
from bim_gw.modules.domain_modules.simple_shapes.attributes import SimpleShapesAttributes
from bim_gw.utils.shapes import generate_dataset
from bim_gw.utils.text_composer.composer import composer
from bim_gw.utils.text_composer.utils import get_choices_from_structure_category, inspect_all_choices
from bim_gw.utils.utils import log_if_save_last_images
from bim_gw.utils.vae import reparameterize


def make_causal_mask_prog(input_dec, encod_out):
    mask = (torch.triu(torch.ones(input_dec.size(1), encod_out.size(1))) == 1).permute(1, 0)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.)).to(input_dec.device)


def convert_angle(angle):
    return angle + 2 * np.pi * (angle < 0)


def symlog(x, alpha=1):
    return torch.sign(x) * torch.log(1 + alpha * torch.abs(x)) / np.log(1 + alpha)


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
            self, z_size, hidden_size, beta, n_classes, imsize, bert_path,
            optim_lr=3e-4, optim_weight_decay=1e-5, scheduler_step=20, scheduler_gamma=0.5,
            domain_examples=None,
            attributes_use_unpaired=True,
            train_vae=True,
            train_attr_decoders=True,
            optimize_vae_with_attr_regression=False,
            coef_attr_loss=1,
            coef_vae_loss=1
    ):

        super(SimpleShapesText, self).__init__()
        self.save_hyperparameters(ignore=["domain_examples"])
        self.n_classes = n_classes
        self.bert_size = 768
        self.hidden_size = hidden_size
        self.z_size = z_size
        self.imsize = imsize
        self.bert_path = bert_path
        self.train_vae = train_vae
        self.train_attr_decoders = train_attr_decoders
        self.optimize_vae_with_attr_regression = optimize_vae_with_attr_regression
        self.coef_attr_loss = coef_attr_loss
        self.coef_vae_loss = coef_vae_loss

        self.text_composer = composer
        self.composer_inspection = inspect_all_choices(composer)

        self.transformer = None
        self.tokenizer = None

        self.attribute_domain = SimpleShapesAttributes(imsize, attributes_use_unpaired)
        self.attribute_domain.freeze()

        self.encoder = nn.Sequential(
            nn.Linear(self.bert_size, self.bert_size),
            nn.ReLU(),
            nn.Linear(self.bert_size, self.bert_size // 2),
            nn.ReLU(),
            nn.Linear(self.bert_size // 2, self.z_size * 2),
            nn.Tanh(),
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
            nn.Linear(self.z_size, sum(self.attribute_domain.output_dims))
        )
        self.grammar_classifiers = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.z_size, self.z_size),
                    nn.ReLU(),
                    nn.Linear(self.z_size, n_outputs)
                )
                for name, n_outputs in self.composer_inspection.items()}
        )

        if not self.train_attr_decoders:
            for param in self.attribute_encoder.parameters():
                param.requires_grad = False
            for param in self.grammar_classifiers.parameters():
                param.requires_grad = False
            self.attribute_encoder.eval()
            self.grammar_classifiers.eval()

        self.grammar_train_acc = nn.ModuleDict(
            {name: torchmetrics.Accuracy() for name in self.composer_inspection.keys()}
        )
        self.grammar_val_acc = nn.ModuleDict(
            {name: torchmetrics.Accuracy() for name in self.composer_inspection.keys()}
        )

        self.domain_examples = domain_examples

        self.output_dims = [self.z_size]
        self.decoder_activation_fn = [
            nn.Tanh(),
        ]

        self.losses = [
            F.mse_loss
        ]

        self.register_buffer("log_sigma", torch.tensor(0.))
        self.register_buffer("beta", torch.tensor(beta))

    def encode(self, sentences):
        if len(sentences) == 3:
            bert_latents, sentences, choices = sentences
            z, _ = self.encode_stats(bert_latents)
            return [z]
        return sentences

    def get_sentence_predictions(self, z, predictions):
        grammar_prediction = self.get_grammar_prediction(z)
        choices = get_choices_from_structure_category(self.text_composer, grammar_prediction)
        cls = predictions[0].detach().cpu().numpy()
        attributes = predictions[1].detach().cpu().numpy()
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
                    "color": (attributes[k, 5] * 255, attributes[k, 6] * 255, attributes[k, 7] * 255),
                    "size": attributes[k, 2],
                    "location": (attributes[k, 0], attributes[k, 1])
                }, choices[k]
            )
            sentence_predictions.append(sentence)
            final_choices.append(choice)
        return sentence_predictions, final_choices

    def decode(self, z):
        z_mean = z[0]
        text_latent = self.decoder(z_mean)
        predictions = self.classify(z_mean)
        predictions = self.attribute_domain.decode(predictions)

        sentence_predictions, final_choices = self.get_sentence_predictions(z_mean, predictions)
        return [text_latent, sentence_predictions, final_choices]

    def sample(self, size, classes=None, min_scale=10, max_scale=25, min_lightness=46, max_lightness=256):
        samples = generate_dataset(size, min_scale, max_scale, min_lightness, max_lightness, 32, classes)
        cls = samples["classes"]
        x, y = samples["locations"][:, 0], samples["locations"][:, 1]
        size = samples["sizes"]
        rotation = samples["rotations"]
        # assert 0 <= rotation <= 1
        # rotation = rotation * 2 * np.pi / 360  # put in radians
        r, g, b = samples["colors"][:, 0], samples["colors"][:, 1], samples["colors"][:, 2]

        labels, choices = self.text_composer(
            {
                "shape": cls,
                "rotation": rotation,
                "color": (r, g, b),
                "size": size,
                "location": (x, y)
            }
        )
        return None, labels, choices  # TODO: add BERT vectors

    def log_domain_from_latent(self, logger, z, name, max_examples=None, step=None):
        predictions = self.attribute_domain.decode(self.classify(z[0]))
        self.attribute_domain.log_domain(logger, predictions, name, max_examples, step=step)

        if logger is not None and hasattr(logger, "log_table"):
            sentences, choices = self.get_sentence_predictions(z[0], predictions)
            text = [[sentence] for sentence in sentences]
            logger.log_table(name + "_s", columns=["Text"], data=text, step=step)

    def log_domain(self, logger, x, name, max_examples=None, step=None):
        self.log_domain_from_latent(logger, self.encode(x), name, max_examples, step)

    def classify(self, z):
        prediction = self.attribute_encoder(z)
        predictions = []
        last_dim = 0
        for dim, act_fn in zip(
                self.attribute_domain.output_dims,
                self.attribute_domain.decoder_activation_fn
        ):
            pred = act_fn(prediction[:, last_dim:last_dim + dim])
            predictions.append(pred)
            last_dim += dim
        return predictions

    def get_grammar_prediction(self, z):
        return {
            name: torch.argmax(self.grammar_classifiers[name](z), dim=1).tolist()
            for name in self.grammar_classifiers.keys()
        }

    def encode_stats(self, text_latent):
        z = self.encoder(text_latent)
        return z[:, :self.z_size], z[:, self.z_size:]

    def reconstruction_loss(self, x_reconstructed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        loss = F.mse_loss(x_reconstructed, x, reduction="sum")
        return loss

    def kl_divergence_loss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return kl

    def step(self, batch, batch_idx, mode="train"):
        sentences, targets = batch["t"][1:], batch["attr"][1:]
        bs = sentences[0].size(0)
        targets = self.attribute_domain.encode(targets)
        x = sentences[0]
        z_mean, z_logvar = self.encode_stats(x)
        z = reparameterize(z_mean, z_logvar)
        x_reconstructed = self.decoder(z)

        reconstruction_loss = self.reconstruction_loss(x_reconstructed, x)
        kl_divergence_loss = self.kl_divergence_loss(z_mean, z_logvar)
        vae_loss = (reconstruction_loss + self.beta * kl_divergence_loss) / bs

        self.log(
            f"{mode}/reconstruction_loss", reconstruction_loss, logger=True, on_epoch=(mode != "train"), batch_size=bs
        )
        self.log(
            f"{mode}/kl_divergence_loss", kl_divergence_loss, logger=True, on_epoch=(mode != "train"), batch_size=bs
        )
        self.log(f"{mode}/vae_loss", vae_loss, on_epoch=(mode != "train"), batch_size=bs)

        z_predictions = z_mean
        if not self.optimize_vae_with_attr_regression:
            z_predictions = z_mean.detach()

        predictions, attribute_losses, attribute_prediction_loss = self.train_attribute_predictions(
            z_predictions, sentences, targets, mode=mode
        )
        total_loss = self.coef_vae_loss * vae_loss + self.coef_attr_loss * attribute_prediction_loss

        self.log(f"{mode}/total_loss", total_loss, on_epoch=(mode != "train"))
        return total_loss

    def train_attribute_predictions(self, x, sentences, targets, mode="train"):
        bs = sentences[0].size(0)
        predictions = self.classify(x)
        losses = []
        total_loss = 0
        for k, (group_pred, loss, target) in enumerate(
                zip(
                    predictions,
                    self.attribute_domain.losses, targets
                )
        ):
            group_loss = loss(group_pred, target)
            predictions.append(group_pred)
            losses.append(group_loss)
            total_loss += group_loss

            self.log(f"{mode}/loss_attributes_{k}", group_loss, logger=True, on_epoch=(mode != "train"), batch_size=bs)

        grammar_coef = 1 / len(self.grammar_classifiers)
        for name, classifier in self.grammar_classifiers.items():
            grammar_prediction = classifier(x.detach())
            loss_grammar = F.cross_entropy(grammar_prediction, sentences[2][name])
            total_loss += grammar_coef * loss_grammar
            acc_fn = self.grammar_train_acc[name] if mode == "train" else self.grammar_val_acc[name]
            res = acc_fn(grammar_prediction.softmax(-1), sentences[2][name])
            self.log(
                f"{mode}/loss_grammar_{name}", loss_grammar, logger=True, on_epoch=(mode != "train"), batch_size=bs
            )
            self.log(f"{mode}/grammar_{name}_acc", res, on_epoch=(mode != "train"))

        self.log(f"{mode}/attribute_grammar_loss", total_loss, on_epoch=(mode != "train"))

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
            domain_examples = self.domain_examples[mode][0]  # only keep in dist
            for logger in self.loggers:
                x = domain_examples["t"][1].to(self.device)
                encoded_s = self.encode(
                    [
                        x,
                        domain_examples["t"][2],
                        domain_examples["t"][3]
                    ]
                )
                predictions = self.classify(encoded_s[0])
                predictions = self.attribute_domain.decode(predictions)
                sentences, choices = self.get_sentence_predictions(encoded_s[0], predictions)

                text = [[sentence] for sentence in sentences]

                if hasattr(logger, "log_table"):
                    logger.log_table(f"{mode}/predictions_text", columns=["Text"], data=text)

                # Images
                self.attribute_domain.log_domain(logger, predictions, f"{mode}/predictions_reconstruction")

                if self.current_epoch == 0:
                    with log_if_save_last_images(logger):
                        self.attribute_domain.log_domain(
                            logger, domain_examples["attr"][1:],
                            f"{mode}/target_reconstruction"
                        )
                        if hasattr(logger, "log_table"):
                            logger.log_table(
                                f"{mode}/target_text", columns=["Text"],
                                data=[[domain_examples['t'][2][k]] for k in
                                      range(len(domain_examples['t'][2]))]
                            )

    def validation_epoch_end(self, outputs):
        self.epoch_end("val")

    def test_epoch_end(self, outputs):
        self.epoch_end("test")

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            params, lr=self.hparams.optim_lr,
            weight_decay=self.hparams.optim_weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.hparams.scheduler_step,
            self.hparams.scheduler_gamma
        )
        return [optimizer], [scheduler]
