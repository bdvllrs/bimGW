import numpy as np
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F

from bim_gw.modules.workspace_module import WorkspaceModule
from bim_gw.utils.losses.losses import nll_loss
from bim_gw.utils.shapes import generate_dataset, log_shape_fig
from bim_gw.utils.text_composer.composer import composer
from bim_gw.utils.text_composer.utils import inspect_all_choices, get_choices_from_structure_category


class ShapesAttributesLM(WorkspaceModule):
    def __init__(self, n_classes, imsize):
        super(ShapesAttributesLM, self).__init__()
        self.save_hyperparameters()

        self.n_classes = n_classes
        self.z_size = 8
        self.imsize = imsize

        self.output_dims = [self.n_classes, self.z_size, 1]
        self.requires_acc_computation = True
        self.decoder_activation_fn = [
            lambda x: torch.log_softmax(x, dim=1),  # shapes
            torch.tanh,  # rest
            torch.tanh  # unpaired
        ]

        self.losses = [
            lambda x, y: nll_loss(x, y),  # shapes
            F.mse_loss,  # rest
            F.mse_loss  # unpaired
        ]

    def encode(self, x):
        if len(x) == 2:
            cls, latents = x
            unpaired = torch.ones_like(latents[:, 0]) * 0.5
        else:
            cls, latents, unpaired = x
        out_latents = latents.clone()
        out_latents[:, 0] = out_latents[:, 0] / self.imsize
        out_latents[:, 1] = out_latents[:, 1] / self.imsize
        out_latents[:, 2] = out_latents[:, 2] / self.imsize
        return (torch.nn.functional.one_hot(cls, self.n_classes).type_as(latents),
                # rotations,
                out_latents * 2 - 1,
                unpaired * 2 - 1)

    def decode(self, x):
        if len(x) == 2:
            logits, latents = x
            unpaired = torch.zeros_like(latents[:, 0])
        else:
            logits, latents, unpaired = x
        out_latents = (latents.clone() + 1) / 2
        out_latents[:, 0] = out_latents[:, 0] * self.imsize
        out_latents[:, 1] = out_latents[:, 1] * self.imsize
        out_latents[:, 2] = out_latents[:, 2] * self.imsize
        return (torch.argmax(logits, dim=-1),
                out_latents,
                (unpaired + 1) / 2)

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


def make_causal_mask_prog(input_dec, encod_out):
    mask = (torch.triu(torch.ones(input_dec.size(1), encod_out.size(1))) == 1).permute(1, 0)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.)).to(input_dec.device)


class ShapesLM(WorkspaceModule):
    def __init__(
            self, z_size, hidden_size, n_classes, imsize, bert_path,
            optim_lr=3e-4, optim_weight_decay=1e-5, scheduler_step=20, scheduler_gamma=0.5,
            domain_examples=None,
    ):

        super(ShapesLM, self).__init__()
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

        self.shapes_attribute = ShapesAttributesLM(n_classes, imsize)
        self.shapes_attribute.freeze()

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
            nn.Linear(self.hidden_size, sum(self.shapes_attribute.output_dims[:-1]))  # remove last unmatched attr
        )
        self.grammar_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.composer_inspection['structures'])  # predict sentence structure
        )

        self.grammar_train_acc = torchmetrics.Accuracy()
        self.grammar_val_acc = torchmetrics.Accuracy()

        self.domain_examples = domain_examples

        self.output_dims = [self.z_size]
        self.decoder_activation_fn = [
            None
        ]

        self.losses = [
            F.mse_loss
        ]
        # self.output_dims = self.shapes_attribute.output_dims
        # self.decoder_activation_fn = self.shapes_attribute.decoder_activation_fn
        # self.losses = self.shapes_attribute.losses

    def encode(self, sentences):
        bert_latents, sentences, choices = sentences
        return [bert_latents]

    def decode(self, text_latent):
        text_latent = text_latent[0]
        z = self.projection(text_latent)
        predictions = self.classify(z)
        grammar_prediction = self.grammar_classifier(z)
        choices = get_choices_from_structure_category(self.text_composer, torch.argmax(grammar_prediction, dim=1).tolist())
        # predictions = text_latent
        predictions = self.shapes_attribute.decode(predictions)
        cls = predictions[0].detach().cpu().numpy()
        attributes = predictions[1].detach().cpu().numpy()
        # Text
        rotation_x = attributes[:, 3] * 2 - 1
        rotation_y = attributes[:, 4] * 2 - 1
        rotations = np.arctan2(rotation_y, rotation_x)

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
        predictions = self.shapes_attribute.decode(self.classify(self.projection(encoded_s[0])))
        # predictions = self.shapes_attribute.decode(encoded_s)
        self.shapes_attribute.log_domain(logger, predictions, name, max_examples, step=step)

    def classify(self, z):
        prediction = self.classifier(z)
        predictions = []
        last_dim = 0
        for dim, act_fn in zip(self.shapes_attribute.output_dims[:-1], self.shapes_attribute.decoder_activation_fn[:-1]):
            pred = act_fn(prediction[:, last_dim:last_dim + dim])
            predictions.append(pred)
            last_dim += dim
        return predictions

    def step(self, batch, batch_idx, mode="train"):
        sentences, targets = batch["t"][1:], batch["attr"][1:]
        bs = sentences[0].size(0)
        targets = self.shapes_attribute.encode(targets)[:-1]
        # if mode == "train":
        #     sentences = (sentences[0] + 0.1 * torch.randn_like(sentences[0]), sentences[1])
        text_latent = self.encode(sentences)[0]
        z = self.projection(text_latent)
        predictions = self.classify(z)
        losses = []
        total_loss = 0
        for k, (group_pred, loss, target) in enumerate(zip(predictions,
                                                           self.shapes_attribute.losses[:-1], targets)):
            group_loss = loss(group_pred, target)
            predictions.append(group_pred)
            losses.append(group_loss)
            total_loss += group_loss

            self.log(f"{mode}/loss_{k}", group_loss, logger=True, on_epoch=(mode != "train"), batch_size=bs)

        grammar_prediction = self.grammar_classifier(z)
        loss_grammar = F.cross_entropy(grammar_prediction, sentences[2])
        total_loss += loss_grammar
        acc_fn = self.grammar_train_acc if mode == "train" else self.grammar_val_acc
        res = acc_fn(grammar_prediction.softmax(-1), sentences[2])
        self.log(f"{mode}/loss_grammar", loss_grammar, logger=True, on_epoch=(mode != "train"), batch_size=bs)
        self.log(f"{mode}_grammar_acc", res, on_epoch=(mode=="val"))

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
                    domain_examples["t"][3].to(self.device)
                ])
                z = self.projection(encoded_s[0])
                predictions = self.classify(z)
                decoded_s = self.decode(encoded_s)
                sentence_predictions = decoded_s[1]

                text = [[sentence_predictions[k]] for k in range(len(sentence_predictions))]

                logger.log_table(f"{mode}/predictions_text", columns=["Text"], data=text)

                # Images
                self.shapes_attribute.log_domain(logger, self.shapes_attribute.decode(predictions),
                                                 f"{mode}/predictions_reconstruction")

                if self.current_epoch == 0:
                    self.shapes_attribute.log_domain(logger, domain_examples["attr"][1:], f"{mode}/target_reconstruction")
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
