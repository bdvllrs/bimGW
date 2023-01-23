import numpy as np
import torch
from PIL import Image

from bim_gw.utils.text_composer.composer import composer
from bim_gw.utils.text_composer.utils import get_categories


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

    def get_items(self, item):
        item = self.get_item(item) if item is not None else self.get_null_item()
        return transform(item, self.transforms)


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


class AttributesDataFetcher(DataFetcher):
    modality = "attr"

    def get_null_item(self):
        # _, cls, attr, _ = self.get_item(0)
        _, cls, attr = self.get_item(0)
        attr[:] = 0.
        # return torch.tensor(0.).float(), 0, attr, torch.tensor(0.).float()
        return torch.tensor(0.).float(), 0, attr

    def get_item(self, item):
        label = self.labels[item]
        cls = int(label[0])
        x, y = label[1], label[2]
        size = label[3]
        rotation = label[4]
        r, g, b = label[5] / 255, label[6] / 255, label[7] / 255
        rotation_x = (np.cos(rotation) + 1) / 2
        rotation_y = (np.sin(rotation) + 1) / 2
        unpaired = label[11]

        return (
            torch.tensor(1.).float(),
            cls,
            torch.tensor([x, y, size, rotation_x, rotation_y, r, g, b, unpaired], dtype=torch.float),
        )


class TextDataFetcher(DataFetcher):
    modality = "t"

    def __init__(self, root_path, split, ids, labels, transforms=None, bert_latents="bert-base-uncased.npy"):
        super(TextDataFetcher, self).__init__(root_path, split, ids, labels, transforms)

        self.bert_data = None
        self.bert_mean = None
        self.bert_std = None
        if bert_latents is not None:
            self.bert_data = np.load(root_path / f"{split}_{bert_latents}")[ids]
            self.bert_mean = np.load(root_path / f"mean_{bert_latents}")
            self.bert_std = np.load(root_path / f"std_{bert_latents}")
            self.bert_data = (self.bert_data - self.bert_mean) / self.bert_std

        self.captions = np.load(str(root_path / f"{split}_captions.npy"))
        self.choices = np.load(str(root_path / f"{split}_caption_choices.npy"), allow_pickle=True)
        self.null_choice = None

    def get_item(self, item):
        sentence = self.captions[item]
        choice = get_categories(composer, self.choices[item])

        if self.transforms is not None:
            sentence = self.transforms(sentence)
        bert = torch.zeros(768).float()
        if self.bert_data is not None:
            bert = torch.from_numpy(self.bert_data[item]).float()
        return torch.tensor(1.).float(), bert, str(sentence), choice

    def get_null_item(self):
        x = torch.zeros(768).float()
        if self.null_choice is None:
            self.null_choice = get_categories(composer, self.choices[0])
            self.null_choice = {key: 0 for key in self.null_choice}
        return torch.tensor(0.).float(), x, "", self.null_choice


class PreSavedLatentDataFetcher:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data[0].shape[0]

    def get_null_item(self):
        return [torch.tensor(0.).float()] + [np.zeros_like(self.data[k][0][0]) for k in range(len(self.data))]

    def get_item(self, item):
        return [torch.tensor(1.).float()] + [self.data[k][item][0] for k in range(len(self.data))]

    def get_items(self, item):
        return self.get_item(item) if item is not None else self.get_null_item()
