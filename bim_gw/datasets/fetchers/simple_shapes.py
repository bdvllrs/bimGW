import numpy as np
import torch
from PIL import Image

from bim_gw.utils.text_composer.composer import Composer
from bim_gw.utils.text_composer.writers import writers


def transform(data, transformation):
    if transformation is not None:
        data = transformation(data)
    return data


class DataFetcher:
    modality = None

    def __init__(self, root_path, split, ids, labels, transforms=None):
        self.root_path = root_path
        self.split = split
        self.ids = ids
        self.labels = labels
        self.transforms = transforms[self.modality]

    def get_null_item(self):
        raise NotImplementedError

    def get_item(self, item):
        raise NotImplementedError

    def get_transformed_item(self, item):
        raise NotImplementedError

    def get_items(self, item, time_steps):
        items = [
            self.get_item(item) if 0 in time_steps else self.get_null_item(),
            self.get_transformed_item(item) if 1 in time_steps else self.get_null_item(),
        ]
        return [transform(item, self.transforms) for item in items]


class VisualDataFetcher(DataFetcher):
    modality = "v"

    def __init__(self, root_path, split, ids, labels, transforms=None):
        super(VisualDataFetcher, self).__init__(root_path, split, ids, labels, transforms)
        self.null_image = None

    def get_null_item(self):
        if self.null_image is None:
            _, x = self.get_item(0)
            shape = list(x.size) + [3]
            img = np.zeros(shape, np.uint8)
            self.null_image = Image.fromarray(img)
        return torch.tensor(0.).float(), self.null_image

    def get_item(self, item, path=None):
        if path is None:
            path = self.root_path

        image_id = self.ids[item]
        with open(path / self.split / f"{image_id}.png", 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return torch.tensor(1.).float(), img

    def get_transformed_item(self, item):
        return self.get_item(item, self.root_path / "transformed")


class AttributesDataFetcher(DataFetcher):
    modality = "attr"

    def get_null_item(self):
        _, cls, attr = self.get_item(0)
        attr[:] = 0.
        return torch.tensor(0.).float(), 0, attr

    def get_transformed_item(self, item):
        label = self.labels[item]
        cls = int(label[11])
        x, y = label[1] + label[12], label[2] + label[13]
        size = label[3] + label[14]
        rotation = label[4] + label[15]
        r, g, b = (label[5] + label[16]) / 255, (label[6] + label[17]) / 255, (label[7] + label[18]) / 255
        rotation_x = (np.cos(rotation) + 1) / 2
        rotation_y = (np.sin(rotation) + 1) / 2

        return (
            torch.tensor(1.).float(),
            cls,
            torch.tensor([x, y, size, rotation_x, rotation_y, r, g, b], dtype=torch.float),
        )

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
            torch.tensor(1.).float(),
            cls,
            torch.tensor([x, y, size, rotation_x, rotation_y, r, g, b], dtype=torch.float),
        )


class TextDataFetcher(DataFetcher):
    modality = "t"

    def __init__(self, root_path, split, ids, labels, transforms=None):
        super(TextDataFetcher, self).__init__(root_path, split, ids, labels, transforms)

        self.sentences = {}
        self.text_composer = Composer(writers)

    def get_item(self, item):
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
        return torch.tensor(1.).float(), sentence

    def get_transformed_item(self, item):
        label = self.labels[item]
        if item in self.sentences:
            sentence = self.sentences[item]
        else:
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
            self.sentences[item] = sentence

        if self.transforms is not None:
            sentence = self.transforms(sentence)
        return torch.tensor(1.).float(), sentence

    def get_null_item(self):
        return torch.tensor(0.).float(), ""


class ActionDataFetcher(DataFetcher):
    modality = "a"

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
            torch.tensor(1.).float(),
            cls,
            torch.tensor([d_x, d_y, d_size, d_rotation_x, d_rotation_y, d_r, d_g, d_b], dtype=torch.float),
        )

    def get_transformed_item(self, item):
        return self.get_item(item)

    def get_null_item(self):
        _, _, x = self.get_item(0)
        x[:] = 0.
        return torch.tensor(0.).float(), 0, x

    def get_items(self, item, time_steps):
        items = [
            self.get_item(item) if 0 in time_steps else self.get_null_item(),
            self.get_transformed_item(item) if 0 in time_steps else self.get_null_item(),
            # 2 times 0 as if it is provided once, it's available at each time step.
        ]
        return [transform(item, self.transforms) for item in items]


class PreSavedLatentDataFetcher:
    def __init__(self, root_path, ids):
        self.root_path = root_path
        self.ids = ids
        self.data = np.load(str(self.root_path))[self.ids]

    def __len__(self):
        return self.data.shape[0]

    def get_null_item(self):
        return torch.tensor(0.).float(), np.zeros_like(self.data[0][0])

    def get_item(self, item):
        return torch.tensor(1.).float(), self.data[item][0]

    def get_transformed_item(self, item):
        return torch.tensor(1.).float(), self.data[item][1]

    def get_items(self, item, time_steps):
        return [
            self.get_item(item) if 0 in time_steps else self.get_null_item(),
            self.get_transformed_item(item) if 1 in time_steps else self.get_null_item(),
        ]
