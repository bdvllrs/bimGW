import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import ArrayLike
from PIL import Image

from bim_gw.datasets.domain import DomainItems
from bim_gw.utils.text_composer.composer import composer
from bim_gw.utils.text_composer.utils import get_categories
from bim_gw.utils.types import SplitLiteral

VisualDataType = Tuple[torch.FloatTensor, Image.Image]
AttributesDataType = Tuple[torch.FloatTensor, int, torch.FloatTensor]
TextDataType = Tuple[torch.FloatTensor, torch.LongTensor, str, Dict[str, int]]


def transform(
    data: DomainItems,
    transformation: Optional[Callable[[DomainItems], DomainItems]]
) -> Any:
    if transformation is not None:
        data = transformation(data)
    return data


class DomainLoader:
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
        self.domain_loader_args = kwargs

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


class VisionLoader(DomainLoader):
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
        super(VisionLoader, self).__init__(
            root_path, split, ids, labels, transforms, **kwargs
        )
        self.null_image = None

    def get_null_item(self) -> DomainItems:
        if self.null_image is None:
            item = self.get_item(0)
            shape = list(item.img.size) + [3]
            img = np.zeros(shape, np.uint8)
            self.null_image = Image.fromarray(img)

        return DomainItems.singular(
            img=self.null_image,
            is_available=False
        )

    def get_item(
        self, item: int, path: Optional[pathlib.Path] = None
    ) -> DomainItems:
        if path is None:
            path = self.root_path

        image_id = self.ids[item]
        with open(path / self.split / f"{image_id}.png", 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        return DomainItems.singular(img=img)


class AttributesLoader(DomainLoader):
    modality = "attr"

    def get_null_item(self) -> DomainItems:
        item = self.get_item(0)

        return DomainItems.singular(
            cls=0,
            attributes=torch.zeros_like(item.attributes),
            is_available=False
        )

    def get_item(self, item: int) -> DomainItems:
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
        if self.domain_loader_args['use_unpaired']:
            attributes.append(unpaired)

        return DomainItems.singular(
            cls=cls,
            attr=torch.tensor(attributes, dtype=torch.float),
        )


_memoize_bert_latents = {}


def _get_bert_latent(file_path: pathlib.Path, **kwargs):
    key = str(file_path.resolve())
    if key in _memoize_bert_latents.keys():
        return _memoize_bert_latents[key]
    _memoize_bert_latents[key] = np.load(file_path, **kwargs)
    return _memoize_bert_latents[key]


class TextLoader(DomainLoader):
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
        super(TextLoader, self).__init__(
            root_path, split, ids, labels, transforms, **kwargs
        )

        if 'bert_latents' not in self.domain_loader_args:
            raise ValueError('bert_latents must be specified for text loader')
        if 'pca_dim' not in self.domain_loader_args:
            raise ValueError('pca_dim must be specified for text loader')

        bert_latents = self.domain_loader_args['bert_latents']
        pca_dim = self.domain_loader_args['pca_dim']

        self.bert_data: Optional[ArrayLike] = None
        self.bert_mean = None
        self.bert_std = None
        if bert_latents is not None:
            if (
                    pca_dim < 768
                    and (root_path / f"{split}_reduced_{pca_dim}_"
                                     f"{bert_latents}").exists()
            ):
                self.bert_data = _get_bert_latent(
                    root_path / f"{split}_reduced_{pca_dim}_{bert_latents}"
                )[ids]
                self.bert_mean = _get_bert_latent(
                    root_path / f"mean_reduced_{pca_dim}_{bert_latents}"
                )
                self.bert_std = _get_bert_latent(
                    root_path / f"std_reduced_{pca_dim}_{bert_latents}"
                )
            elif pca_dim == 768:
                self.bert_data = _get_bert_latent(
                    root_path / f"{split}_{bert_latents}"
                )[ids]
                self.bert_mean = _get_bert_latent(
                    root_path / f"mean_{bert_latents}"
                )
                self.bert_std = _get_bert_latent(
                    root_path / f"std_{bert_latents}"
                )
            else:
                raise ValueError("No PCA data found")
            self.bert_data: np.ndarray = (
                    (self.bert_data - self.bert_mean) / self.bert_std
            )

        self.captions = _get_bert_latent(
            root_path / f"{split}_captions.npy"
        )
        self.choices = _get_bert_latent(
            root_path / f"{split}_caption_choices.npy", allow_pickle=True
        )
        self.null_choice = None

    def get_item(self, item: int) -> DomainItems:
        sentence = self.captions[item]
        choice = get_categories(composer, self.choices[item])

        if self.transforms is not None:
            sentence = self.transforms(sentence)
        bert = torch.zeros(768).float()
        if self.bert_data is not None:
            bert = torch.from_numpy(self.bert_data[item]).float()

        return DomainItems.singular(
            bert=bert,
            text=sentence,
            choices=choice
        )

    def get_null_item(self) -> DomainItems:
        if self.null_choice is None:
            self.null_choice = get_categories(composer, self.choices[0])
            self.null_choice = {key: 0 for key in self.null_choice}

        return DomainItems.singular(
            bert=torch.zeros(768).float(),
            text="",
            choices=self.null_choice,
            is_available=False
        )


class PreSavedLatentLoader:
    def __init__(self, data, domain_item_mapping):
        self.data = data
        self.domain_item_mapping = domain_item_mapping

    def __len__(self) -> int:
        return self.data[0].shape[0]

    def _get_items(self, item):
        return {
            self.domain_item_mapping[k]: np.zeros_like(self.data[k][item][0])
            for k in range(len(self.data))
        }

    def get_null_item(self):
        return DomainItems.singular(
            **self._get_items(0),
            is_available=False,
        )

    def get_item(self, item: int):
        return DomainItems.singular(
            **self._get_items(item),
        )

    def get_items(self, item: int):
        return self.get_item(
            item
        ) if item is not None else self.get_null_item()
