import numpy as np
import torch
from gensim.models import KeyedVectors
from torch.nn import functional as F
from transformers import BartForConditionalGeneration, BartTokenizer

from bim_gw.modules.workspace_module import WorkspaceModule
from bim_gw.utils.losses.losses import nll_loss
from bim_gw.utils.shapes import generate_dataset, log_shape_fig


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

        labels = [
            torch.from_numpy(cls),
            torch.from_numpy(np.stack([x, y, radius, rotation_x, rotation_y, r, g, b], axis=1)).to(torch.float),
        ]
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
        text = ", ".join(map(str, [classes[0].item()] + latents[0].tolist()))
        logger.experiment[name + "_text"].log(text)


def make_causal_mask_prog(input_dec, encod_out):
    mask = (torch.triu(torch.ones(input_dec.size(1), encod_out.size(1))) == 1).permute(1, 0)
    return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.)).to(input_dec.device)

class ShapesLM(WorkspaceModule):
    def __init__(self, n_classes, imsize):
        super(ShapesLM, self).__init__()
        self.n_classes = n_classes
        self.z_size = 3
        self.imsize = imsize

        self.bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

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

    def decode(self, text_latent):
        device = text_latent.device
        text_latent = text_latent
        text_dec = torch.ones(text_latent.size(0), text_latent.size(1) + 2,
                              dtype=torch.long, device=device).fill_(2)
        logits = torch.zeros(text_latent.size(0), text_latent.size(1), self.bart.model.shared.weight.size(0))
        text_dec[:, 1] = 0
        text_dec[:, -1] = 2

        for i in range(text_latent.size(1) - 1):
            mask = make_causal_mask_prog(text_dec[:, :-2], text_latent)

            decod_out, = self.bart.model.decoder(text_dec[:, :-2], text_latent, encoder_padding_mask=None,
                                                 decoder_padding_mask=None,
                                                 decoder_causal_mask=mask)

            logits[:, i + 1] = F.linear(decod_out,
                                        self.bart.model.shared.weight,
                                        self.bart.final_logits_bias)[:, i + 1]

            if i + 2 < text_dec.size(1):
                text_dec[:, i + 2] = torch.argmax(logits[:, i + 1], dim=1)
        return text_dec[:, 1:]

    def forward(self, x):
        tokens = self.bart_tokenizer(x[0], return_tensors='pt', padding=True)['input_ids'].to(self.device)
        encoding = self.bart.model.encoder(tokens)[0]
        return encoding

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

        labels = [
            torch.from_numpy(cls),
            torch.from_numpy(np.stack([x, y, radius, rotation_x, rotation_y, r, g, b], axis=1)).to(torch.float),
        ]
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
        text = ", ".join(map(str, [classes[0].item()] + latents[0].tolist()))
        logger.experiment[name + "_text"].log(text)
