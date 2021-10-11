import numpy as np
import torch
from gensim.models import KeyedVectors

from bim_gw.modules.workspace_module import WorkspaceModule
from bim_gw.utils.losses.losses import nll_loss
from bim_gw.utils.shapes import generate_dataset, log_shape_fig

from torch.nn import functional as F

class SkipGramLM(WorkspaceModule):
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


class ShapesLM(WorkspaceModule):
    def __init__(self, n_classes, imsize):
        super(ShapesLM, self).__init__()
        self.n_classes = n_classes
        self.z_size = 3
        self.imsize = imsize

        self.output_dims = [self.z_size, 7]
        self.requires_acc_computation = True
        self.decoder_activation_fn = [
            lambda x: torch.log_softmax(x, dim=1),  # shapes
            # torch.tanh,  # rotations
            torch.sigmoid,  # rest
        ]

        self.losses = [
            nll_loss,  # shapes
            # F.mse_loss,  # rotations
            F.mse_loss  # rest
        ]

    def encode(self, x):
        return self(x)

    def decode(self, x):
        logits, latent = x
        out_latents = latent.clone()
        out_latents[:, 0] = latent[:, 0] * self.imsize
        out_latents[:, 1] = latent[:, 1] * self.imsize
        out_latents[:, 2] = latent[:, 2] * self.imsize
        return (torch.argmax(logits, dim=-1),
                out_latents)

    def forward(self, x: list):
        cls, latents = x
        out_latents = latents.clone()
        out_latents[:, 0] = latents[:, 0] / self.imsize
        out_latents[:, 1] = latents[:, 1] / self.imsize
        out_latents[:, 2] = latents[:, 2] / self.imsize
        return (torch.nn.functional.one_hot(cls, self.n_classes).type_as(latents),
                # rotations,
                out_latents)

    def compute_acc(self, acc_metric, predictions, targets):
        return acc_metric(predictions[0], targets[0].to(torch.int16))

    def log_domain(self, logger, x, name, max_examples=None):
        classes = x[0][:max_examples].detach().cpu().numpy()
        # rotations = x[1][:max_examples].detach().cpu().numpy()
        # rotation_x = rotations[:, 0]
        # rotation_y = rotations[:, 1]
        latents = x[1][:max_examples].detach().cpu().numpy()

        # from cos / sin coordinates to angle in degrees
        # rotations = (360 + np.arctan2(rotation_y, rotation_x) / np.pi * 180) % 360

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
