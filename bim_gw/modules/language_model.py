import numpy as np
import torch
from gensim.models import KeyedVectors

from bim_gw.modules.workspace_module import WorkspaceModule


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
    def __init__(self, n_classes):
        super(ShapesLM, self).__init__()
        self.n_classes = n_classes
        self.z_size = 3

    def encode(self, x):
        return self(x)

    def decode(self, x):
        logits, _ = x
        return torch.softmax(logits, dim=-1)

    def forward(self, x):
        cls, latents = x
        return torch.nn.functional.one_hot(cls, self.n_classes).type_as(latents), latents

    def get_targets(self, targets):
        return targets[0].to(torch.int16)

    def get_random_vector(self, classes):
        z = torch.rand(classes.size(0), 7).to(classes.device)
        return z


