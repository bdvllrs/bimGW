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
        self.transforms = transforms["a"]

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
        self.text_composer = Composer(writers)

    def __getitem__(self, item):
        label = self.labels[item]
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

        if self.transforms is not None:
            sentence = self.transforms(sentence)
        return sentence


class TransformationDataFetcher:
    def __init__(self, root_path, split, ids, labels, transforms=None):
        self.labels = labels
        self.transforms = transforms["action"]

    def __getitem__(self, item):
        label = self.labels[item]
        cls = int(label[11])
        d_x, d_y = label[12], label[13]
        d_size = label[14]
        d_rotation = label[15]
        d_r, d_g, d_b = label[16] / 255, label[17] / 255, label[18] / 255
        d_rotation_x = (np.cos(d_rotation) + 1) / 2
        d_rotation_y = (np.sin(d_rotation) + 1) / 2

        labels = (
            cls,
            torch.tensor([d_x, d_y, d_size, d_rotation_x, d_rotation_y, d_r, d_g, d_b], dtype=torch.float),
        )

        if self.transforms is not None:
            labels = self.transforms(labels)
        return labels


class TransformedVisualDataFetcher:
    def __init__(self, root_path, split, ids, labels, transforms=None):
        self.data_fetcher = VisualDataFetcher(root_path / "transformed", split, ids, labels, transforms)

    def __getitem__(self, item):
        return self.data_fetcher[item]


class TransformedAttributesDataFetcher(AttributesDataFetcher):
    def __getitem__(self, item):
        label = self.labels[item]
        cls = int(label[11])
        x, y = label[1] + label[12], label[2] + label[13]
        size = label[3] + label[14]
        rotation = label[4] + label[15]
        r, g, b = (label[5] + label[16]) / 255, (label[6] + label[17]) / 255, (label[7] + label[18]) / 255
        rotation_x = (np.cos(rotation) + 1) / 2
        rotation_y = (np.sin(rotation) + 1) / 2

        labels = (
            cls,
            torch.tensor([x, y, size, rotation_x, rotation_y, r, g, b], dtype=torch.float),
        )

        if self.transforms is not None:
            labels = self.transforms(labels)
        return labels


class TransformedTextDataFetcher(TextDataFetcher):
    def __getitem__(self, item):
        label = self.labels[item]
        cls = int(label[11])
        x, y = label[1] + label[12], label[2] + label[13]
        size = label[3] + label[14]
        rotation = label[4] + label[15]
        r, g, b = label[5] + label[16], label[6] + label[17], label[7] + label[18]

        sentence = self.text_composer({
            "shape": cls,
            "rotation": rotation,
            "color": (r, g, b),
            "size": size,
            "location": (x, y)
        })

        if self.transforms is not None:
            sentence = self.transforms(sentence)
        return sentence
