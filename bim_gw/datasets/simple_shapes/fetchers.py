import numpy as np
import torch
from PIL import Image

from bim_gw.utils.text_composer.composer import Composer
from bim_gw.utils.text_composer.writers import writers


class VisualDataFetcher:
    def __init__(self, root_path, split, ids, labels, transforms=None):
        self.root_path = root_path
        self.split = split
        self.ids = ids
        self.transforms = transforms["v"]

    def __getitem__(self, item):
        image_id = self.ids[item]
        with open(self.root_path / self.split / f"{image_id}.png", 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


class AttributesDataFetcher:
    def __init__(self, root_path, split, ids, labels, transforms=None):
        self.labels = labels
        self.transforms = transforms["attr"]

    def __getitem__(self, item):
        label = self.labels[item]
        cls = int(label[0])
        x, y = label[1], label[2]
        size = label[3]
        rotation = label[4]
        r, g, b = label[5] / 255, label[6] / 255, label[7] / 255
        rotation_x = (np.cos(rotation) + 1) / 2
        rotation_y = (np.sin(rotation) + 1) / 2

        labels = (
            cls,
            torch.tensor([x, y, size, rotation_x, rotation_y, r, g, b], dtype=torch.float),
        )

        if self.transforms is not None:
            labels = self.transforms(labels)
        return labels


class TextDataFetcher:
    def __init__(self, root_path, split, ids, labels, transforms=None):
        self.labels = labels
        self.transforms = transforms["t"]
        self.sentences = {}
        self.text_composer = Composer(writers)

    def __getitem__(self, item):
        label = self.labels[item]
        if item in self.sentences:
            sentence = self.sentences[item]
        else:
            cls = int(label[0])
            x, y = label[1], label[2]
            size = label[3]
            rotation = label[4]

            sentence = self.text_composer({
                "shape": cls,
                "rotation": rotation,
                "color": (label[5], label[6], label[7]),
                "size": size,
                "location": (x, y)
            })
            self.sentences[item] = sentence

        if self.transforms is not None:
            sentence = self.transforms(sentence)
        return sentence


class PreSavedLatentDataFetcher:
    def __init__(self, root_path, ids):
        self.root_path = root_path
        self.ids = ids
        self.data = np.load(str(self.root_path))[self.ids]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]
