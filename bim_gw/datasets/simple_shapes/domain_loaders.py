import pathlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from bim_gw.datasets.domain import DomainItems
from bim_gw.datasets.simple_shapes.types import (
    AvailableDomainsType,
    TransformType
)
from bim_gw.utils.text_composer.composer import composer
from bim_gw.utils.text_composer.utils import get_categories
from bim_gw.utils.types import SplitLiteral


def transform(
    data: DomainItems,
    transformation: Optional[Callable[[DomainItems], DomainItems]]
) -> Any:
    if transformation is not None:
        data = transformation(data)
    return data


class DomainLoader:
    modality: AvailableDomainsType

    def __init__(
        self,
        root_path: pathlib.Path,
        split: SplitLiteral,
        ids: np.ndarray,
        labels,
        transforms: Optional[Dict[AvailableDomainsType, Optional[
            TransformType]]] = None,
        **kwargs
    ):
        self.root_path = root_path
        self.split = split
        self.ids = ids
        self.labels = labels
        self.transforms: Optional[TransformType] = None
        if transforms is not None:
            self.transforms = transforms[self.modality]
        self.domain_loader_args = kwargs

    def get_null_item(self) -> DomainItems:
        raise NotImplementedError

    def get_item(self, item: int) -> DomainItems:
        raise NotImplementedError

    def get_items(self, item: int) -> DomainItems:
        selected_item = self.get_item(
            item
        ) if item is not None else self.get_null_item()
        return transform(selected_item, self.transforms)


class VisionLoader(DomainLoader):
    modality: AvailableDomainsType = "v"

    def __init__(
        self,
        root_path: pathlib.Path,
        split: SplitLiteral,
        ids: np.ndarray,
        labels,
        transforms: Optional[
            Dict[AvailableDomainsType, Optional[TransformType]]] =
        None,
        **kwargs
    ):
        super(VisionLoader, self).__init__(
            root_path, split, ids, labels, transforms, **kwargs
        )
        self.null_image = None

    def get_null_item(self) -> DomainItems:
        if self.null_image is None:
            item = self.get_item(0)
            shape = list(item['img'].size) + [3]
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
    modality: AvailableDomainsType = "attr"

    def __init__(
        self,
        root_path: pathlib.Path,
        split: SplitLiteral,
        ids: np.ndarray,
        labels,
        transforms: Optional[
            Dict[AvailableDomainsType, Optional[TransformType]]] =
        None,
        **kwargs
    ):
        super(AttributesLoader, self).__init__(
            root_path, split, ids, labels, transforms, **kwargs
        )

        self.attributes, self.cls = self.get_attributes()
        self.zero_cls = torch.tensor(0, dtype=torch.long)
        self.zero_attr = torch.zeros_like(self.attributes[0])

    def get_attributes(self):
        x, y = self.labels[:, 1], self.labels[:, 2]
        size = self.labels[:, 3]
        rotation = self.labels[:, 4]
        r = self.labels[:, 5] / 255
        g = self.labels[:, 6] / 255
        b = self.labels[:, 7] / 255
        rotation_x = (np.cos(rotation) + 1) / 2
        rotation_y = (np.sin(rotation) + 1) / 2
        unpaired = self.labels[:, 11]
        attributes = [x, y, size, rotation_x, rotation_y, r, g, b]
        if self.domain_loader_args['use_unpaired']:
            attributes.append(unpaired)
        return torch.from_numpy(
            np.stack(
                attributes, axis=1
            )
        ).to(torch.float), torch.from_numpy(self.labels[:, 0]).to(torch.long)

    def get_null_item(self) -> DomainItems:
        return DomainItems.singular(
            cls=self.zero_cls,
            attr=self.zero_attr,
            is_available=False
        )

    def get_item(self, item: int) -> DomainItems:
        return DomainItems.singular(
            cls=self.cls[item],
            attr=self.attributes[item],
        )


_memoize_bert_latents: Dict[str, np.ndarray] = {}


def _get_bert_latent(file_path: pathlib.Path, **kwargs) -> np.ndarray:
    key = str(file_path.resolve())
    if key in _memoize_bert_latents.keys():
        return _memoize_bert_latents[key]
    _memoize_bert_latents[key] = np.load(file_path, **kwargs)
    return _memoize_bert_latents[key]


class TextLoader(DomainLoader):
    modality: AvailableDomainsType = "t"

    def __init__(
        self,
        root_path: pathlib.Path,
        split: SplitLiteral,
        ids: np.ndarray,
        labels,
        transforms: Optional[
            Dict[AvailableDomainsType, Optional[TransformType]]] = None,
        **kwargs
    ):
        super(TextLoader, self).__init__(
            root_path, split, ids, labels, transforms, **kwargs
        )

        if 'bert_latents' not in self.domain_loader_args:
            raise ValueError('bert_latents must be specified for text loader')
        if 'pca_dim' not in self.domain_loader_args:
            raise ValueError('pca_dim must be specified for text loader')

        self.bert_data: Optional[torch.Tensor] = None
        self.bert_mean: Optional[torch.Tensor] = None
        self.bert_std: Optional[torch.Tensor] = None

        if self.domain_loader_args['bert_latents'] is not None:
            bert_data, bert_mean, bert_std = self._get_bert_data(
                ids, root_path, split
            )
            self.bert_data = torch.from_numpy(bert_data).float()
            self.bert_mean = torch.from_numpy(bert_mean).float()
            self.bert_std = torch.from_numpy(bert_std).float()

        self.captions = _get_bert_latent(
            root_path / f"{split}_captions.npy"
        )
        self.choices = _get_bert_latent(
            root_path / f"{split}_caption_choices.npy", allow_pickle=True
        )

        # Load first for self.null_choice, the next ones are lazy-loaded
        self.categories = {
            0: get_categories(composer, self.choices[0])
        }

        self.null_bert = torch.zeros(768).float()
        self.null_choice = {
            key: 0
            for key in self.categories[0].keys()
        }

    def _get_bert_data(
        self, ids: np.ndarray, root_path: pathlib.Path, split: SplitLiteral
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bert_latents = self.domain_loader_args['bert_latents']
        pca_dim = self.domain_loader_args['pca_dim']

        if bert_latents is None:
            raise ValueError('bert_latents must be specified for text loader')
        if (
                pca_dim < 768
                and (root_path / f"{split}_reduced_{pca_dim}_"
                                 f"{bert_latents}").exists()
        ):
            bert_data = _get_bert_latent(
                root_path / f"{split}_reduced_{pca_dim}_{bert_latents}"
            )[ids]
            bert_mean = _get_bert_latent(
                root_path / f"mean_reduced_{pca_dim}_{bert_latents}"
            )
            bert_std = _get_bert_latent(
                root_path / f"std_reduced_{pca_dim}_{bert_latents}"
            )
        elif pca_dim == 768:
            bert_data = _get_bert_latent(
                root_path / f"{split}_{bert_latents}"
            )[ids]
            bert_mean = _get_bert_latent(
                root_path / f"mean_{bert_latents}"
            )
            bert_std = _get_bert_latent(
                root_path / f"std_{bert_latents}"
            )
        else:
            raise ValueError("No PCA data found")
        return (bert_data - bert_mean) / bert_std, bert_mean, bert_std

    def get_item(self, item: int) -> DomainItems:
        sentence = self.captions[item]

        if self.transforms is not None:
            sentence = self.transforms(sentence)

        bert = self.null_bert
        if self.bert_data is not None:
            bert = self.bert_data[item]

        if item not in self.categories.keys():
            # Lazy-load the categories
            self.categories[item] = get_categories(
                composer, self.choices[item]
            )

        return DomainItems.singular(
            bert=bert,
            text=sentence,
            choices=self.categories[item]
        )

    def get_null_item(self) -> DomainItems:
        return DomainItems.singular(
            bert=self.null_bert,
            text="",
            choices=self.null_choice,
            is_available=False
        )


class PreSavedLatentLoader:
    def __init__(
        self, data: List[np.ndarray], domain_item_mapping
    ):
        self.data = [torch.from_numpy(data[k]) for k in range(len(data))]
        self.domain_item_mapping = domain_item_mapping
        self._null_item = self._get_null_item()

    def __len__(self) -> int:
        return self.data[0].shape[0]

    def _get_items(self, item):
        return {
            self.domain_item_mapping[k]: self.data[k][item][0]
            for k in range(len(self.data))
        }

    def _get_null_item(self):
        return {
            k: torch.zeros_like(v)
            for k, v in self._get_items(0).items()
        }

    def get_null_item(self):
        return DomainItems.singular(
            **self._null_item,
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
