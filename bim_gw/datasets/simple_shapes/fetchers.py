import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import ArrayLike
from PIL import Image

from bim_gw.utils.text_composer.composer import composer
from bim_gw.utils.text_composer.utils import get_categories
from bim_gw.utils.types import SplitLiteral

VisualDataType = Tuple[torch.FloatTensor, Image.Image]
AttributesDataType = Tuple[torch.FloatTensor, int, torch.FloatTensor]
TextDataType = Tuple[torch.FloatTensor, torch.LongTensor, str, Dict[str, int]]


def transform(
    data: Any, transformation: Optional[Callable[[Any], Any]]
) -> Any:
    if transformation is not None:
        data = transformation(data)
    return data


class DataFetcher:
    modality: str = None

    def __init__(
        self,
        root_path: pathlib.Path,
        split: SplitLiteral,
        ids: List[int],
        labels,
        transforms: Optional[Dict[str, Callable[[Any], Any]]] = None,
        **kwargs
    ):
        self.root_path = root_path
        self.split = split
        self.ids = ids
        self.labels = labels
        self.transforms = transforms[self.modality]
        self.fetcher_args = kwargs

    def get_null_item(self):
        raise NotImplementedError

    def get_item(self, item: int):
        raise NotImplementedError

    def get_items(
        self, item: int
    ) -> Union[VisualDataType, AttributesDataType, TextDataType]:
        item = self.get_item(
            item
        ) if item is not None else self.get_null_item()
        return transform(item, self.transforms)


class VisualDataFetcher(DataFetcher):
    modality = "v"

    def __init__(
        self,
        root_path: pathlib.Path,
        split: SplitLiteral,
        ids: List[int],
        labels,
        transforms: Optional[Dict[str, Callable[[Any], Any]]] = None,
        **kwargs
    ):
        super(VisualDataFetcher, self).__init__(
            root_path, split, ids, labels, transforms, **kwargs
        )
        self.null_image = None

    def get_null_item(self) -> VisualDataType:
        if self.null_image is None:
            _, x = self.get_item(0)
            shape = list(x.size) + [3]
            img = np.zeros(shape, np.uint8)
            self.null_image = Image.fromarray(img)
        return torch.tensor(0.).float(), self.null_image

    def get_item(
        self, item: int, path: Optional[pathlib.Path] = None
    ) -> VisualDataType:
        if path is None:
            path = self.root_path

        image_id = self.ids[item]
        with open(path / self.split / f"{image_id}.png", 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return torch.tensor(1.).float(), img


class AttributesDataFetcher(DataFetcher):
    modality = "attr"

    def get_null_item(self) -> AttributesDataType:
        # _, cls, attr, _ = self.get_item(0)
        _, cls, attr = self.get_item(0)
        attr[:] = 0.
        # return torch.tensor(0.).float(), 0, attr, torch.tensor(0.).float()
        return torch.tensor(0.).float(), 0, attr

    def get_item(self, item: int) -> AttributesDataType:
        label = self.labels[item]
        cls = int(label[0])
        x, y = label[1], label[2]
        size = label[3]
        rotation = label[4]
        r, g, b = label[5] / 255, label[6] / 255, label[7] / 255
        rotation_x = (np.cos(rotation) + 1) / 2
        rotation_y = (np.sin(rotation) + 1) / 2
        unpaired = label[11]

        attributes = [x, y, size, rotation_x, rotation_y, r, g, b]
        if self.fetcher_args['use_unpaired']:
            attributes.append(unpaired)

        return (
            torch.tensor(1.).float(),
            cls,
            torch.tensor(attributes, dtype=torch.float),
        )


class TextDataFetcher(DataFetcher):
    modality = "t"

    def __init__(
        self,
        root_path: pathlib.Path,
        split: SplitLiteral,
        ids: List[int],
        labels,
        transforms: Optional[Dict[str, Callable[[Any], Any]]] = None,
        **kwargs
    ):
        super(TextDataFetcher, self).__init__(
            root_path, split, ids, labels, transforms, **kwargs
        )

        assert 'bert_latents' in self.fetcher_args, 'bert_latents must be ' \
                                                    'specified for text ' \
                                                    'fetcher'
        assert 'pca_dim' in self.fetcher_args, 'pca_dim must be specified ' \
                                               'for text fetcher'

        bert_latents = self.fetcher_args['bert_latents']
        pca_dim = self.fetcher_args['pca_dim']

        self.bert_data: Optional[ArrayLike] = None
        self.bert_mean = None
        self.bert_std = None
        if bert_latents is not None:
            if (
                    pca_dim < 768
                    and (root_path / f"{split}_reduced_{pca_dim}_"
                                     f"{bert_latents}").exists()
            ):
                self.bert_data = np.load(
                    root_path / f"{split}_reduced_{pca_dim}_{bert_latents}"
                )[ids]
                self.bert_mean = np.load(
                    root_path / f"mean_reduced_{pca_dim}_{bert_latents}"
                )
                self.bert_std = np.load(
                    root_path / f"std_reduced_{pca_dim}_{bert_latents}"
                )
            elif pca_dim == 768:
                self.bert_data = np.load(
                    root_path / f"{split}_{bert_latents}"
                )[ids]
                self.bert_mean = np.load(root_path / f"mean_{bert_latents}")
                self.bert_std = np.load(root_path / f"std_{bert_latents}")
            else:
                raise ValueError("No PCA data found")
            self.bert_data: np.ndarray = (
                    (self.bert_data - self.bert_mean) / self.bert_std
            )

        self.captions = np.load(str(root_path / f"{split}_captions.npy"))
        self.choices = np.load(
            str(root_path / f"{split}_caption_choices.npy"), allow_pickle=True
        )
        self.null_choice = None

    def get_item(self, item: int) -> TextDataType:
        sentence = self.captions[item]
        choice = get_categories(composer, self.choices[item])

        if self.transforms is not None:
            sentence = self.transforms(sentence)
        bert = torch.zeros(768).float()
        if self.bert_data is not None:
            bert = torch.from_numpy(self.bert_data[item]).float()
        return torch.tensor(1.).float(), bert, str(sentence), choice

    def get_null_item(self) -> TextDataType:
        x = torch.zeros(768).float()
        if self.null_choice is None:
            self.null_choice = get_categories(composer, self.choices[0])
            self.null_choice = {key: 0 for key in self.null_choice}
        return torch.tensor(0.).float(), x, "", self.null_choice


class PreSavedLatentDataFetcher:
    def __init__(self, data):
        self.data = data

    def __len__(self) -> int:
        return self.data[0].shape[0]

    def get_null_item(self):
        return [torch.tensor(0.).float()] + [np.zeros_like(self.data[k][0][0])
                                             for k in range(len(self.data))]

    def get_item(self, item: int):
        return [torch.tensor(1.).float()] + [self.data[k][item][0] for k in
                                             range(len(self.data))]

    def get_items(self, item: int):
        return self.get_item(
            item
        ) if item is not None else self.get_null_item()
