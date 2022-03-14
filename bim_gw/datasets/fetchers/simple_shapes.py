import numpy as np
import torch
from PIL import Image

from bim_gw.utils.text_composer.composer import Composer
from bim_gw.utils.text_composer.writers import writers


def transform(data, transformation):
    if transformation is not None:
        data = transformation(data)
    return data


class VisualDataFetcher:
    def __init__(self, root_path, split, ids, labels, transforms=None):
        self.root_path = root_path
        self.split = split
        self.ids = ids
        self.transforms = transforms["v"]
        self.null_image = None

    def get_null_item(self):
        if self.null_image is None:
            _, x = self.get_item(0)
            shape = list(x.size) + [3]
            img = np.zeros(shape, np.uint8)
            self.null_image = Image.fromarray(img)
        return transform((0, self.null_image), self.transforms)

    def get_item(self, item):
        image_id = self.ids[item]
        with open(self.root_path / self.split / f"{image_id}.png", 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return (1, img)

    def __getitem__(self, item):
        return transform(self.get_item(item), self.transforms)


class AttributesDataFetcher:
    def __init__(self, root_path, split, ids, labels, transforms=None):
        self.labels = labels
        self.transforms = transforms["attr"]

    def get_null_item(self):
        _, cls, attr = self.get_item(0)
        attr[:] = 0.
        return transform((0, 0, attr), self.transforms)

    def get_item(self, item):
        label = self.labels[item]
        cls = int(label[0])
        x, y = label[1], label[2]
        size = label[3]
        rotation = label[4]
        r, g, b = label[5] / 255, label[6] / 255, label[7] / 255
        rotation_x = (np.cos(rotation) + 1) / 2
        rotation_y = (np.sin(rotation) + 1) / 2

        return (
            1,
            cls,
            torch.tensor([x, y, size, rotation_x, rotation_y, r, g, b], dtype=torch.float),
        )

    def __getitem__(self, item):
        return transform(self.get_item(item), self.transforms)


class TextDataFetcher:
    def __init__(self, root_path, split, ids, labels, transforms=None):
        self.labels = labels
        self.transforms = transforms["t"]
        self.text_composer = Composer(writers)

    def get_item(self, item):
        label = self.labels[item]
        cls = int(label[0])
        x, y = label[1], label[2]
        size = label[3]
        rotation = label[4]

        return (1, self.text_composer({
            "shape": cls,
            "rotation": rotation,
            "color": (label[5], label[6], label[7]),
            "size": size,
            "location": (x, y)
        }))

    def get_null_item(self):
        return transform((0, ""), self.transforms)

    def __getitem__(self, item):
        return transform(self.get_item(item), self.transforms)


class TransformationDataFetcher:
    def __init__(self, root_path, split, ids, labels, transforms=None):
        self.labels = labels
        self.transforms = transforms["a"]

    def get_item(self, item):
        label = self.labels[item]
        orig_cls = int(label[0])
        cls = int(label[11])
        # predict 4 classes: no transformation (label 0), or transform to 1 of 3 classes
        cls = 0 if cls == orig_cls else int(label[11]) + 1
        d_x, d_y = label[12], label[13]
        d_size = label[14]
        d_rotation = label[15]
        d_r, d_g, d_b = label[16] / 255, label[17] / 255, label[18] / 255
        d_rotation_x = np.cos(d_rotation)
        d_rotation_y = np.sin(d_rotation)

        return (
            1,
            cls,
            torch.tensor([d_x, d_y, d_size, d_rotation_x, d_rotation_y, d_r, d_g, d_b], dtype=torch.float),
        )

    def get_null_item(self):
        _, _, x = self.get_item(0)
        x[:] = 0.
        return transform((0, 0, x), self.transforms)

    def __getitem__(self, item):
        return transform(self.get_item(item), self.transforms)


class TransformedVisualDataFetcher:
    def __init__(self, root_path, split, ids, labels, transforms=None):
        self.data_fetcher = VisualDataFetcher(root_path / "transformed", split, ids, labels, transforms)

    def get_null_item(self):
        return self.data_fetcher.get_null_item()

    def get_item(self, item):
        return self.data_fetcher.get_item(item)

    def __getitem__(self, item):
        return self.data_fetcher[item]


class TransformedAttributesDataFetcher(AttributesDataFetcher):
    def get_item(self, item):
        label = self.labels[item]
        cls = int(label[11])
        x, y = label[1] + label[12], label[2] + label[13]
        size = label[3] + label[14]
        rotation = label[4] + label[15]
        r, g, b = (label[5] + label[16]) / 255, (label[6] + label[17]) / 255, (label[7] + label[18]) / 255
        rotation_x = (np.cos(rotation) + 1) / 2
        rotation_y = (np.sin(rotation) + 1) / 2

        return (
            1,
            cls,
            torch.tensor([x, y, size, rotation_x, rotation_y, r, g, b], dtype=torch.float),
        )


class TransformedTextDataFetcher(TextDataFetcher):
    def get_item(self, item):
        label = self.labels[item]
        cls = int(label[11])
        x, y = label[1] + label[12], label[2] + label[13]
        size = label[3] + label[14]
        rotation = label[4] + label[15]
        r, g, b = label[5] + label[16], label[6] + label[17], label[7] + label[18]

        return (1, self.text_composer({
            "shape": cls,
            "rotation": rotation,
            "color": (r, g, b),
            "size": size,
            "location": (x, y)
        }))


class PreSavedLatentDataFetcher:
    def __init__(self, root_path, ids):
        self.root_path = root_path
        self.ids = ids
        self.data = np.load(str(self.root_path))[self.ids]

    def __len__(self):
        return self.data.shape[0]

    def get_null_item(self):
        return (0, np.zeros_like(self.data[0]))

    def get_item(self, item):
        return (1, self.data[item])

    def __getitem__(self, item):
        return self.get_item(item)
