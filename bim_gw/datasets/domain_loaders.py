import pathlib
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import numpy as np
import torch

from bim_gw.datasets.domain import DomainItems, TransformType
from bim_gw.utils.types import AvailableDomains, SplitLiteral


def transform(
    data: DomainItems,
    transformation: Optional[Callable[[DomainItems], DomainItems]],
) -> Any:
    if transformation is not None:
        data = transformation(data)
    return data


class DomainLoader:
    modality: AvailableDomains

    def __init__(
        self,
        root_path: pathlib.Path,
        split: SplitLiteral,
        ids: np.ndarray,
        labels,
        transforms: Optional[
            Mapping[AvailableDomains, Optional[TransformType]]
        ] = None,
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

    def get_items(self, item: Optional[int]) -> DomainItems:
        selected_item = (
            self.get_item(item) if item is not None else self.get_null_item()
        )
        return transform(selected_item, self.transforms)


class PreSavedLatentLoader:
    modality = "pre_saved"

    def __init__(
        self, data: List[np.ndarray], domain_item_mapping: Dict[int, str]
    ):
        self.data = [torch.from_numpy(data[k]) for k in range(len(data))]
        self.domain_item_mapping = domain_item_mapping
        self._null_item = self._get_null_item()

    def __len__(self) -> int:
        return self.data[0].shape[0]

    def _get_items(self, item: int) -> Dict[str, torch.Tensor]:
        return {
            self.domain_item_mapping[k]: self.data[k][item][0]
            for k in range(len(self.data))
        }

    def _get_null_item(self) -> Dict[str, torch.Tensor]:
        return {k: torch.zeros_like(v) for k, v in self._get_items(0).items()}

    def get_null_item(self) -> DomainItems:
        return DomainItems.singular(
            **self._null_item,
            is_available=False,
        )

    def get_item(self, item: int) -> DomainItems:
        return DomainItems.singular(
            **self._get_items(item),
        )

    def get_items(self, item: Optional[int]) -> DomainItems:
        return (
            self.get_item(item) if item is not None else self.get_null_item()
        )


DomainLoaderType = Union[DomainLoader, PreSavedLatentLoader]
