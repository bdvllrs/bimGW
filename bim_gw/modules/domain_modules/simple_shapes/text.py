import numpy as np
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F

from bim_gw.modules.domain_modules.simple_shapes.attributes import SimpleShapesAttributes
from bim_gw.modules.domain_modules.domain_module import DomainModule
from bim_gw.modules.workspace_encoders import DomainEncoder, DomainDecoder
from bim_gw.utils.shapes import generate_dataset
from bim_gw.utils.text_composer.composer import composer
from bim_gw.utils.text_composer.utils import inspect_all_choices, get_choices_from_structure_category


def make_causal_mask_prog(input_dec, encod_out):
    mask = (torch.triu(torch.ones(input_dec.size(1), encod_out.size(1))) == 1).permute(1, 0)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.)).to(input_dec.device)


def convert_angle(angle):
    return angle + 2 * np.pi * (angle < 0)


class SimpleShapesText(DomainModule):
    def __init__(
            self, z_size, hidden_size, n_classes, imsize, bert_path,
            optim_lr=3e-4, optim_weight_decay=1e-5, scheduler_step=20, scheduler_gamma=0.5,
            domain_examples=None,
    ):

        super(SimpleShapesText, self).__init__()
        self.save_hyperparameters(ignore=["domain_examples"])
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.z_size = z_size
        self.imsize = imsize
        self.bert_path = bert_path

        self.text_composer = composer
        self.composer_inspection = inspect_all_choices(composer)

        self.transformer = None
        self.tokenizer = None

        self.attribute_domain = SimpleShapesAttributes(imsize)
        self.attribute_domain.freeze()

        self.projection = nn.Sequential(
            nn.Linear(self.z_size, self.z_size),
            nn.ReLU(),
            nn.Linear(self.z_size, self.z_size // 2),
            nn.ReLU(),
            nn.Linear(self.z_size // 2, self.hidden_size)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, sum(self.attribute_domain.output_dims))  # remove last unmatched attr
        )
        self.grammar_classifiers = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, n_outputs))  # predict sentence structure
            for name, n_outputs in self.composer_inspection.items()})

        self.grammar_train_acc = nn.ModuleDict(
            {name: torchmetrics.Accuracy() for name in self.composer_inspection.keys()})
        self.grammar_val_acc = nn.ModuleDict(
            {name: torchmetrics.Accuracy() for name in self.composer_inspection.keys()})

        self.domain_examples = domain_examples

        self.output_dims = [self.z_size]
        self.decoder_activation_fn = [
            None
        ]

        self.losses = [
            F.mse_loss
        ]
        self.workspace_encoder_cls = DomainEncoder
        self.workspace_decoder_cls = DomainDecoder
        # self.output_dims = self.attribute_domain.output_dims
        # self.decoder_activation_fn = self.attribute_domain.decoder_activation_fn
        # self.losses = self.attribute_domain.losses

    def encode(self, sentences):
        bert_latents, sentences, choices = sentences
        return [bert_latents]

    def decode(self, text_latent):
        text_latent = text_latent[0]
        z = self.projection(text_latent)
        predictions = self.classify(z)
        grammar_prediction = self.get_grammar_prediction(z)
        choices = get_choices_from_structure_category(self.text_composer, grammar_prediction)
        # predictions = text_latent
        predictions = self.attribute_domain.decode(predictions)
        cls = predictions[0].detach().cpu().numpy()
        attributes = predictions[1].detach().cpu().numpy()
        # Text
        rotation_x = attributes[:, 3] * 2 - 1
        rotation_y = attributes[:, 4] * 2 - 1
        rotations = convert_angle(np.arctan2(rotation_y, rotation_x))

        sentence_predictions, final_choices = [], []
        for k in range(len(cls)):
            sentence, choice = self.text_composer({
                "shape": int(cls[k]),
                "rotation": rotations[k],
                "color": (attributes[k, 5] * 255, attributes[k, 6] * 255, attributes[k, 7] * 255),
                "size": attributes[k, 2],
                "location": (attributes[k, 0], attributes[k, 1])
            }, choices[k])
            sentence_predictions.append(sentence)
            final_choices.append(choice)
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

        labels, choices = self.text_composer({
            "shape": cls,
            "rotation": rotation,
            "color": (r, g, b),
            "size": size,
            "location": (x, y)
        })
        return None, labels, choices  # TODO: add BERT vectors

    def log_domain(self, logger, x, name, max_examples=None, step=None):
        if logger is not None:
            text = [[x[1][k]] for k in range(len(x[1]))]
            logger.log_table(name + "_s", columns=["Text"], data=text, step=step)
        if type(x[0]) == list:
            encoded_s = x[0]
        else:
            encoded_s = self.encode(x)
        predictions = self.attribute_domain.decode(self.classify(self.projection(encoded_s[0])))
        # predictions = self.attribute_domain.decode(encoded_s)
        self.attribute_domain.log_domain(logger, predictions, name, max_examples, step=step)

    def classify(self, z):
        prediction = self.classifier(z)
        predictions = []
        last_dim = 0
        for dim, act_fn in zip(self.attribute_domain.output_dims,
                               self.attribute_domain.decoder_activation_fn):
            pred = act_fn(prediction[:, last_dim:last_dim + dim])
            predictions.append(pred)
            last_dim += dim
        return predictions

    def get_grammar_prediction(self, z):
        return {
            name: torch.argmax(self.grammar_classifiers[name](z), dim=1).tolist()
            for name in self.grammar_classifiers.keys()
        }

    def step(self, batch, batch_idx, mode="train"):
        sentences, targets = batch["t"][1:], batch["attr"][1:]
        bs = sentences[0].size(0)
        targets = self.attribute_domain.encode(targets)
        # if mode == "train":
        #     sentences = (sentences[0] + 0.1 * torch.randn_like(sentences[0]), sentences[1])
        text_latent = self.encode(sentences)[0]
        z = self.projection(text_latent)
        predictions = self.classify(z)
        losses = []
        total_loss = 0
        for k, (group_pred, loss, target) in enumerate(zip(predictions,
                                                           self.attribute_domain.losses, targets)):
            group_loss = loss(group_pred, target)
            predictions.append(group_pred)
            losses.append(group_loss)
            total_loss += group_loss

            self.log(f"{mode}/loss_{k}", group_loss, logger=True, on_epoch=(mode != "train"), batch_size=bs)

        grammar_coef = 1 / len(self.grammar_classifiers)
        for name, classifier in self.grammar_classifiers.items():
            grammar_prediction = classifier(z)
            loss_grammar = F.cross_entropy(grammar_prediction, sentences[2][name])
            total_loss += grammar_coef * loss_grammar
            acc_fn = self.grammar_train_acc[name] if mode == "train" else self.grammar_val_acc[name]
            res = acc_fn(grammar_prediction.softmax(-1), sentences[2][name])
            self.log(f"{mode}/loss_grammar_{name}", loss_grammar, logger=True, on_epoch=(mode != "train"),
                     batch_size=bs)
            self.log(f"{mode}_grammar_{name}_acc", res, on_epoch=(mode != "train"))

        self.log(f"{mode}/total_loss", total_loss, logger=True, on_epoch=(mode != "train"), batch_size=bs)
        return predictions, losses, total_loss

    def training_step(self, batch, batch_idx):
        predictions, losses, total_loss = self.step(batch, batch_idx, "train")
        return total_loss

    def validation_step(self, batch, batch_idx):
        predictions, losses, total_loss = self.step(batch, batch_idx, "val")
        return total_loss

    def test_step(self, batch, batch_idx):
        predictions, losses, total_loss = self.step(batch, batch_idx, "test")
        return total_loss

    def epoch_end(self, mode="val"):
        if self.domain_examples is not None and mode in self.domain_examples:
            domain_examples = self.domain_examples[mode][0]  # only keep in dist
            for logger in self.loggers:
                encoded_s = self.encode([
                    domain_examples["t"][1].to(self.device),
                    domain_examples["t"][2],
                    domain_examples["t"][3]
                ])
                z = self.projection(encoded_s[0])
                predictions = self.classify(z)
                decoded_s = self.decode(encoded_s)
                sentence_predictions = decoded_s[1]

                text = [[sentence_predictions[k]] for k in range(len(sentence_predictions))]

                logger.log_table(f"{mode}/predictions_text", columns=["Text"], data=text)

                # Images
                self.attribute_domain.log_domain(logger, self.attribute_domain.decode(predictions),
                                                 f"{mode}/predictions_reconstruction")

                if self.current_epoch == 0:
                    self.attribute_domain.log_domain(logger, domain_examples["attr"],
                                                     f"{mode}/target_reconstruction")
                    logger.log_table(f"{mode}/target_text", columns=["Text"],
                                     data=[[domain_examples['t'][2][k]] for k in
                                           range(len(domain_examples['t'][2]))])

    def validation_epoch_end(self, outputs):
        self.epoch_end("val")

    def test_epoch_end(self, outputs):
        self.epoch_end("test")

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.hparams.optim_lr,
                                     weight_decay=self.hparams.optim_weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.scheduler_step,
                                                    self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]
