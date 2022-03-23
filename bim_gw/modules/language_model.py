import numpy as np
import torch
from gensim.models import KeyedVectors
from torch import nn
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel

from bim_gw.modules.workspace_module import WorkspaceModule
from bim_gw.utils.losses.losses import nll_loss
from bim_gw.utils.shapes import generate_dataset, log_shape_fig
from bim_gw.utils.text_composer.composer import Composer
from bim_gw.utils.text_composer.writers import writers


class SkipGramLM(WorkspaceModule):
    def log_domain(self, logger, x, title, max_examples=None):
        pass

    def compute_acc(self, acc_metric, predictions, targets):
        pass

    def __init__(self, path, classnames, load_embeddings=None):
        super(SkipGramLM, self).__init__()

        if load_embeddings is None:
            self.gensim_model = KeyedVectors.load_word2vec_format(path)

            unavailable_classes = []
            embeddings = []
            for classname in classnames:
                unavailable = True
                for cls in classname:
                    embs = []
                    try:
                        emb = self.gensim_model.get_vector(cls.lower().replace(" ", "_"))
                        embs.append(emb)
                    except KeyError:
                        try:
                            emb = self.gensim_model.get_vector(cls.lower().replace("-", "_"))
                            embs.append(emb)
                        except KeyError:
                            cls = cls.replace("-", " ")
                            for word in cls.split(" "):
                                try:
                                    embs.append(self.gensim_model.get_vector(word.lower()))
                                except KeyError:
                                    print(f"{word} does not have an embedding, in {classname}.")
                    if len(embs):
                        embeddings.append(np.vstack(embs).mean(axis=0))
                        unavailable = False
                        break
                if unavailable:
                    unavailable_classes.append(classname)

            word_vectors = torch.from_numpy(np.vstack(embeddings))
        else:
            word_vectors = torch.from_numpy(np.load(load_embeddings, allow_pickle=True))

        self.z_size = word_vectors.size(1)

        self.register_buffer("word_vectors", word_vectors)

    def encode(self, targets):
        return self(targets)

    def decode(self, z):
        return z @ self.word_vectors.t()

    def forward(self, targets):
        embeddings = self.word_vectors.gather(0, targets[:, None].expand(-1, self.z_size))
        return embeddings

    def get_targets(self, targets):
        return targets

    def get_random_vector(self, classes):
        return classes


class ShapesAttributesLM(WorkspaceModule):
    def __init__(self, n_classes, imsize):
        super(ShapesAttributesLM, self).__init__()
        self.n_classes = n_classes
        self.z_size = 3
        self.imsize = imsize

        self.output_dims = [self.z_size, 8]
        self.requires_acc_computation = True
        self.decoder_activation_fn = [
            lambda x: torch.softmax(x, dim=1),  # shapes
            # torch.tanh,  # rotations
            torch.tanh,  # rest
        ]

        self.losses = [
            lambda x, y: nll_loss(x.log(), y),  # shapes
            # F.mse_loss,  # rotations
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

    def log_domain(self, logger, x, name, max_examples=None):
        classes = x[0][:max_examples].detach().cpu().numpy()
        latents = x[1][:max_examples].detach().cpu().numpy()

        # visualization
        log_shape_fig(
            logger,
            classes,
            # rotations,
            latents,
            name + "_vis"
        )

        # text
        labels = ["x", "y", "s", "rotx", "roty", "r", "g", "b"]
        for k in range(len(classes)):
            text = f"c: {str(classes[k].item())}, " + ", ".join(map(lambda a: f"{a[0]}: {a[1]:.4f}", zip(labels, latents[k].tolist())))
            logger.experiment[name + "_text"].log(f"{k+1} - {text}")
        logger.experiment[name + "_text"].log("-----")


def make_causal_mask_prog(input_dec, encod_out):
    mask = (torch.triu(torch.ones(input_dec.size(1), encod_out.size(1))) == 1).permute(1, 0)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.)).to(input_dec.device)


class ShapesLM(WorkspaceModule):
    def __init__(
            self, z_size, n_classes, imsize, bert_path,
            optim_lr=3e-4, optim_weight_decay=1e-5, scheduler_step=20, scheduler_gamma=0.5,
            validation_domain_examples=None,
    ):

        super(ShapesLM, self).__init__()
        self.save_hyperparameters()
        self.n_classes = n_classes
        self.z_size = z_size
        self.bert_size = 768
        self.imsize = imsize

        self.transformer =  BertModel.from_pretrained(bert_path)
        for p in self.transformer.parameters():
            p.requires_grad_(False)

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        self.text_composer = Composer(writers)

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

        self.validation_domain_examples = validation_domain_examples

        self.output_dims = [self.z_size]
        self.decoder_activation_fn = [
            None
        ]

        self.losses = [
            F.mse_loss
        ]

    def encode(self, x):
        return self(x)

    def decode(self, text_latent):
        predictions = self.classify(text_latent)
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
        return sentence_predictions

    def get_bert_latent(self, sentences):
        tokens = self.tokenizer(sentences, return_tensors='pt', padding=True).to(self.device)
        x = self.transformer(**tokens)["last_hidden_state"][:, 0]
        return x

    def forward(self, sentences):
        bert_latent = self.get_bert_latent(sentences)
        return self.projection(bert_latent)

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

    def log_domain(self, logger, x, name, max_examples=None):
        for k in range(len(x)):
            logger.experiment[name + "_s"].log(f"{k + 1} - {x[k]}")
        logger.experiment[name + "_s"].log("-----")
        encoded_s = self.encode(x)
        predictions = self.shapes_attribute.decode(self.classify(encoded_s))
        self.shapes_attribute.log_domain(logger, predictions, name, max_examples)

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
        sentences, targets = batch["t"], batch["a"]
        bs = len(sentences)
        targets = self.shapes_attribute.encode(targets)
        z = self.encode(sentences)
        predictions = self.classify(z)
        losses = []
        total_loss = 0
        for k, (group_pred, loss, target) in enumerate(zip(predictions,
                                                           self.shapes_attribute.losses, targets)):
            group_loss = loss(group_pred, target)
            predictions.append(group_pred)
            losses.append(group_loss)
            total_loss += group_loss

            self.log(f"{mode}_loss_{k}", group_loss, logger=True, on_epoch=(mode == "val"), batch_size=bs)

        self.log(f"{mode}_total_loss", total_loss, logger=True, on_epoch=(mode == "val"), batch_size=bs)
        return predictions, losses, total_loss

    def training_step(self, batch, batch_idx):
        predictions, losses, total_loss = self.step(batch["sync_"], batch_idx, "train")
        return total_loss

    def validation_step(self, batch, batch_idx):
        predictions, losses, total_loss = self.step(batch, batch_idx, "val")
        return total_loss

    def validation_epoch_end(self, outputs):
        if self.logger is not None:
            encoded_s = self.encode(self.validation_domain_examples["t"])
            predictions = self.classify(encoded_s)
            sentence_predictions = self.decode(encoded_s)

            for k in range(len(sentence_predictions)):
                self.logger.experiment["predictions_text"].log(f"{k+1} - {sentence_predictions[k]}")
            self.logger.experiment["predictions_text"].log("-----")

            # Images
            self.shapes_attribute.log_domain(self.logger, self.shapes_attribute.decode(predictions),
                                             "predictions_reconstruction")

            if self.current_epoch == 0:
                self.shapes_attribute.log_domain(self.logger, self.validation_domain_examples["a"], "target_reconstruction")
                for k in range(len(sentence_predictions)):
                    self.logger.experiment["target_text"].log(f"{k+1} - {self.validation_domain_examples['t'][k]}")

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.hparams.optim_lr,
                                     weight_decay=self.hparams.optim_weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.scheduler_step,
                                                    self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]
