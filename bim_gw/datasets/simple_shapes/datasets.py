import pathlib
from pathlib import Path
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
)

import numpy as np
from torch.utils.data import Dataset

from bim_gw.datasets.domain import DomainItems
from bim_gw.datasets.domain_loaders import (
    DomainLoader,
    DomainLoaderType,
    PreSavedLatentLoader,
)
from bim_gw.datasets.pre_saved_latents import load_pre_saved_latent
from bim_gw.datasets.simple_shapes.domain_loaders import (
    AttributesLoader,
    TextLoader,
    VisionLoader,
)
from bim_gw.datasets.simple_shapes.types import (
    SelectedDomainType,
    ShapesAvailableDomains,
)
from bim_gw.utils.types import AvailableDomains, SequenceLike, SplitLiteral

AVAILABLE_DOMAINS: Dict[ShapesAvailableDomains, Type[DomainLoader]] = {
    ShapesAvailableDomains.v: VisionLoader,
    ShapesAvailableDomains.attr: AttributesLoader,
    ShapesAvailableDomains.t: TextLoader,
}

domain_item_name_mapping: Dict[ShapesAvailableDomains, Dict[int, str]] = {
    ShapesAvailableDomains.v: {0: "z_img"},
    ShapesAvailableDomains.attr: {0: "z_cls", 1: "z_attr"},
    ShapesAvailableDomains.t: {0: "z"},
}


class SimpleShapesDataset(Dataset):
    def __init__(
        self,
        path: Union[str, pathlib.Path],
        split: SplitLiteral = "train",
        mapping: Optional[Sequence[int]] = None,
        domain_mapping: Optional[
            Sequence[Sequence[ShapesAvailableDomains]]
        ] = None,
        selected_indices: Optional[SequenceLike] = None,
        selected_domains: Optional[Sequence[ShapesAvailableDomains]] = None,
        pre_saved_latent_path: Optional[
            Dict[ShapesAvailableDomains, str]
        ] = None,
        transform: Optional[
            Mapping[ShapesAvailableDomains, Optional[Callable[[Any], Any]]]
        ] = None,
        output_transform: Optional[
            Callable[[SelectedDomainType], SelectedDomainType]
        ] = None,
        domain_loader_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            path:
            split:
            transform:
            output_transform:
            selected_indices:
        """
        assert split in ["train", "val", "test"]
        self.selected_domains = selected_domains or list(
            AVAILABLE_DOMAINS.keys()
        )
        self.root_path = Path(path)
        self.transforms: Optional[
            Mapping[ShapesAvailableDomains, Optional[Callable[[Any], Any]]]
        ] = {
            domain: (
                transform[domain]
                if (transform is not None and domain in transform)
                else None
            )
            for domain in AVAILABLE_DOMAINS.keys()
        }
        self.output_transform = output_transform
        self.split = split
        self.img_size = 32

        self.classes = np.array(["square", "circle", "triangle"])
        self.labels: np.ndarray = np.load(
            str(self.root_path / f"{split}_labels.npy")
        )
        self.ids: np.ndarray = np.arange(len(self.labels))
        if selected_indices is not None:
            self.labels = self.labels[selected_indices]
            self.ids = self.ids[selected_indices]

        self.selected_indices = (
            selected_indices
            if selected_indices is not None
            else np.arange(len(self.labels))
        )

        self.mapping: np.ndarray = np.array(mapping or self.ids)
        self.available_domains_mapping = domain_mapping or (
            [list(self.selected_domains)] * self.mapping.shape[0]
        )

        if domain_loader_params is None:
            domain_loader_params = dict()
        for domain in AVAILABLE_DOMAINS.keys():
            if domain not in domain_loader_params:
                domain_loader_params[domain] = dict()

        self._domain_loader_params = domain_loader_params
        self.domain_loaders: Dict[ShapesAvailableDomains, DomainLoaderType] = {
            domain: AVAILABLE_DOMAINS[domain](
                self.root_path,
                self.split,
                self.ids,
                self.labels,
                self.transforms,  # type: ignore
                **self._domain_loader_params[domain],
            )
            for domain in self.selected_domains
        }

        self.pre_saved_latent_path = pre_saved_latent_path
        self.pre_saved_data: Dict[
            ShapesAvailableDomains, List[np.ndarray]
        ] = {}
        if pre_saved_latent_path is not None:
            self.use_pre_saved_latents(pre_saved_latent_path)

    def subset(self, idx: Sequence[int]) -> "SimpleShapesDataset":
        mapping = [self.mapping[k] for k in idx]
        domain_mapping = [self.available_domains_mapping[k] for k in idx]
        selected_idx = [k for k in idx if k in self.selected_indices]

        return SimpleShapesDataset(
            self.root_path,
            self.split,
            mapping,
            domain_mapping,
            selected_idx,
            self.selected_domains,
            self.pre_saved_latent_path,
            self.transforms,
            self.output_transform,
            self._domain_loader_params,
        )

    def use_pre_saved_latents(
        self, pre_saved_latent_path: Dict[ShapesAvailableDomains, str]
    ) -> None:
        for domain_key in self.selected_domains:
            if domain_key in pre_saved_latent_path.keys():
                self.pre_saved_data[domain_key] = load_pre_saved_latent(
                    self.root_path,
                    self.split,
                    cast(Dict[AvailableDomains, str], pre_saved_latent_path),
                    domain_key,
                    self.ids,
                )
                self.domain_loaders[domain_key] = PreSavedLatentLoader(
                    self.pre_saved_data[domain_key],
                    domain_item_name_mapping[domain_key],
                )

    def __len__(self) -> int:
        return self.mapping.shape[0]

    def __getitem__(self, item: int) -> SelectedDomainType:
        idx = self.mapping[item]
        mapping = self.available_domains_mapping[item]

        selected_domains: Dict[ShapesAvailableDomains, DomainItems] = {}
        n_domains = 0

        for domain_key, domain_loader in self.domain_loaders.items():
            if domain_key in mapping:
                domain_items = domain_loader.get_items(idx)
            else:
                domain_items = domain_loader.get_items(None)
            n_domains += domain_items.available_masks.item()
            selected_domains[domain_key] = domain_items
        assert n_domains == len(mapping)
        if self.output_transform is not None:
            return self.output_transform(selected_domains)
        return selected_domains
