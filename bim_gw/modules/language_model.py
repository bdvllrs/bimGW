import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel, BertTokenizerFast, BertTokenizer

from bim_gw.modules.workspace_module import WorkspaceModule
from bim_gw.utils.losses.losses import nll_loss
from bim_gw.utils.shapes import generate_dataset, log_shape_fig
from bim_gw.utils.text_composer.composer import composer


class ShapesAttributesLM(WorkspaceModule):
    def __init__(self, n_classes, imsize):
        super(ShapesAttributesLM, self).__init__()
        self.save_hyperparameters()

        self.n_classes = n_classes
        self.z_size = 8
        self.imsize = imsize

        self.output_dims = [self.n_classes, self.z_size]
        self.requires_acc_computation = True
        self.decoder_activation_fn = [
            lambda x: torch.log_softmax(x, dim=1),  # shapes
            torch.tanh,  # rest
        ]

        self.losses = [
            lambda x, y: nll_loss(x, y),  # shapes
            F.mse_loss  # rest
        ]

    def encode(self, x):
        return self(x)

    def decode(self, x):
        logits, latent = x
        out_latents = (latent.clone() + 1) / 2
        out_latents[:, 0] = out_latents[:, 0] * self.imsize
        out_latents[:, 1] = out_latents[:, 1] * self.imsize
        out_latents[:, 2] = out_latents[:, 2] * self.imsize
        return (torch.argmax(logits, dim=-1),
                out_latents)

    def adapt(self, x):
        return x[0].exp(), x[1]

    def forward(self, x: list):
        cls, latents = x
        out_latents = latents.clone()
        out_latents[:, 0] = out_latents[:, 0] / self.imsize
        out_latents[:, 1] = out_latents[:, 1] / self.imsize
        out_latents[:, 2] = out_latents[:, 2] / self.imsize
        return (torch.nn.functional.one_hot(cls, self.n_classes).type_as(latents),
                # rotations,
                out_latents * 2 - 1)

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
        text = []
        for k in range(len(classes)):
            text.append([classes[k].item()] + latents[k].tolist())
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
            self, z_size, n_classes, imsize, bert_path,
            optim_lr=3e-4, optim_weight_decay=1e-5, scheduler_step=20, scheduler_gamma=0.5,
            domain_examples=None,
    ):

        super(ShapesLM, self).__init__()
        self.save_hyperparameters(ignore=["domain_examples"])
        self.n_classes = n_classes
        self.z_size = z_size
        self.bert_size = 768
        assert self.z_size != self.bert_size, "Cannot have z_size and bert_size the same."
        self.imsize = imsize
        self.bert_path = bert_path

        self.text_composer = composer

        self.transformer = None
        self.tokenizer = None

        self.shapes_attribute = ShapesAttributesLM(n_classes, imsize)
        self.shapes_attribute.freeze()

        self.projection = nn.Sequential(
            nn.Linear(self.bert_size, self.bert_size),
            nn.ReLU(),
            nn.Linear(self.bert_size, self.bert_size // 2),
            nn.ReLU(),
            nn.Linear(self.bert_size // 2, self.z_size)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.z_size, self.z_size),
            nn.ReLU(),
            nn.Linear(self.z_size, sum(self.shapes_attribute.output_dims))
        )

        self.domain_examples = domain_examples

        # self.output_dims = [self.z_size]
        # self.decoder_activation_fn = [
        #     None
        # ]
        #
        # self.losses = [
        #     F.mse_loss
        # ]
        self.output_dims = self.shapes_attribute.output_dims
        self.decoder_activation_fn = self.shapes_attribute.decoder_activation_fn
        self.losses = self.shapes_attribute.losses

    def encode(self, x):
        return self(x)

    def decode(self, text_latent):
        # text_latent = text_latent[0]
        # predictions = self.classify(text_latent)
        predictions = text_latent
        predictions = self.shapes_attribute.decode(predictions)
        cls = predictions[0].detach().cpu().numpy()
        attributes = predictions[1].detach().cpu().numpy()
        # Text
        rotation_x = attributes[:, 3] * 2 - 1
        rotation_y = attributes[:, 4] * 2 - 1
        rotations = np.arctan2(rotation_y, rotation_x)

        sentence_predictions = [self.text_composer({
            "shape": int(cls[k]),
            "rotation": rotations[k],
            "color": (attributes[k, 5] * 255, attributes[k, 6] * 255, attributes[k, 7] * 255),
            "size": attributes[k, 2],
            "location": (attributes[k, 0], attributes[k, 1])
        }) for k in range(len(cls))]
        return [text_latent, sentence_predictions]

    def get_bert_latent(self, sentences):
        if self.transformer is None:
            self.transformer = BertModel.from_pretrained(self.bert_path)
            self.transformer.eval()
            self.transformer.to(self.device)
            for p in self.transformer.parameters():
                p.requires_grad_(False)
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        tokens = self.tokenizer(sentences, return_tensors='pt', padding=True).to(self.device)
        x = self.transformer(**tokens)["last_hidden_state"][:, 0]
        return x

    def forward(self, sentences):
        bert_latents, sentences = sentences
        return self.classify(self.projection(bert_latents))

    def sample(self, size, classes=None, min_scale=10, max_scale=25, min_lightness=46, max_lightness=256):
        samples = generate_dataset(size, min_scale, max_scale, min_lightness, max_lightness, 32, classes)
        cls = samples["classes"]
        x, y = samples["locations"][:, 0], samples["locations"][:, 1]
        size = samples["sizes"]
        rotation = samples["rotations"]
        # assert 0 <= rotation <= 1
        # rotation = rotation * 2 * np.pi / 360  # put in radians
        r, g, b = samples["colors"][:, 0], samples["colors"][:, 1], samples["colors"][:, 2]

        labels = self.text_composer({
            "shape": cls,
            "rotation": rotation,
            "color": (r, g, b),
            "size": size,
            "location": (x, y)
        })
        return labels

    def log_domain(self, logger, x, name, max_examples=None, step=None):
        if logger is not None:
            text = [[x[1][k]] for k in range(len(x[1]))]
            logger.log_table(name + "_s", columns=["Text"], data=text, step=step)
        if type(x[0]) == list:
            encoded_s = x[0]
        else:
            encoded_s = self.encode(x)
        # predictions = self.shapes_attribute.decode(self.classify(encoded_s))
        predictions = self.shapes_attribute.decode(encoded_s)
        self.shapes_attribute.log_domain(logger, predictions, name, max_examples, step=step)

    def classify(self, z):
        prediction = self.classifier(z)
        predictions = []
        last_dim = 0
        for dim, act_fn in zip(self.shapes_attribute.output_dims, self.shapes_attribute.decoder_activation_fn):
            pred = act_fn(prediction[:, last_dim:last_dim + dim])
            predictions.append(pred)
            last_dim += dim
        return predictions

    def step(self, batch, batch_idx, mode="train"):
        sentences, targets = batch["t"][1:], batch["a"][1:]
        bs = sentences[0].size(0)
        targets = self.shapes_attribute.encode(targets)
        # if mode == "train":
        #     sentences = (sentences[0] + 0.1 * torch.randn_like(sentences[0]), sentences[1])
        z = self.encode(sentences)[0]
        predictions = self.classify(z)
        losses = []
        total_loss = 0
        for k, (group_pred, loss, target) in enumerate(zip(predictions,
                                                           self.shapes_attribute.losses, targets)):
            group_loss = loss(group_pred, target)
            predictions.append(group_pred)
            losses.append(group_loss)
            total_loss += group_loss

            self.log(f"{mode}/loss_{k}", group_loss, logger=True, on_epoch=(mode != "train"), batch_size=bs)

        self.log(f"{mode}/total_loss", total_loss, logger=True, on_epoch=(mode != "train"), batch_size=bs)
        return predictions, losses, total_loss

    def training_step(self, batch, batch_idx):
        batch, target = batch
        predictions, losses, total_loss = self.step(batch, batch_idx, "train")
        return total_loss

    def validation_step(self, batch, batch_idx):
        batch, target = batch
        predictions, losses, total_loss = self.step(batch, batch_idx, "val")
        return total_loss

    def test_step(self, batch, batch_idx):
        batch, target = batch
        predictions, losses, total_loss = self.step(batch, batch_idx, "test")
        return total_loss

    def epoch_end(self, mode="val"):
        if self.domain_examples is not None and mode in self.domain_examples:
            domain_examples = self.domain_examples[mode][0][0]  # only keep in dist
            for logger in self.loggers:
                encoded_s = self.encode([
                    domain_examples["t"][1].to(self.device),
                    domain_examples["t"][2]
                ])
                predictions = self.classify(encoded_s[0])
                sentence_predictions = self.decode(encoded_s)[1]

                text = [[sentence_predictions[k]] for k in range(len(sentence_predictions))]

                logger.log_table(f"{mode}/predictions_text", columns=["Text"], data=text)

                # Images
                self.shapes_attribute.log_domain(logger, self.shapes_attribute.decode(predictions),
                                                 f"{mode}/predictions_reconstruction")

                if self.current_epoch == 0:
                    self.shapes_attribute.log_domain(logger, domain_examples["a"][1:], f"{mode}/target_reconstruction")
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
