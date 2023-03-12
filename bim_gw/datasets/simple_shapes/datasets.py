import pathlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

import numpy as np
from numpy.typing import ArrayLike

from bim_gw.datasets.pre_saved_latents import load_pre_saved_latent
from bim_gw.datasets.simple_shapes.fetchers import (
    AttributesDataFetcher, AttributesDataType, DataFetcher, PreSavedLatentDataFetcher, TextDataFetcher,
    TextDataType, VisualDataFetcher, VisualDataType
)
from bim_gw.utils.types import SplitLiteral

AvailableDomainsType = Literal["v", "attr", "t"]

SelectedDomainType = Dict[str, Union[VisualDataType, AttributesDataType, TextDataType]]


class SimpleShapesDataset:
    available_domains: Dict[AvailableDomainsType, Type[DataFetcher]] = {
        "v": VisualDataFetcher,
        "attr": AttributesDataFetcher,
        "t": TextDataFetcher,
    }

    def __init__(
            self,
            path: Union[str, pathlib.Path],
            split: SplitLiteral = "train",
            mapping: List[int] = None,
            domain_mapping: List[List[str]] = None,
            selected_indices: Optional[ArrayLike[int]] = None,
            selected_domains: List[AvailableDomainsType] = None,
            pre_saved_latent_path: Optional[Dict[str, str]] = None,
            transform: Optional[Dict[AvailableDomainsType, Callable[[Any], Any]]] = None,
            output_transform: Optional[Callable[[SelectedDomainType], SelectedDomainType]] = None,
            fetcher_params: Optional[Dict[str, Any]] = None
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
        self.selected_domains = [domain for domain in
                                 self.available_domains.keys()] if selected_domains is None else selected_domains
        self.root_path = Path(path)
        self.transforms = {domain: (transform[domain] if (transform is not None and domain in transform) else None)
                           for domain in self.available_domains.keys()}
        self.output_transform = output_transform
        self.split = split
        self.img_size = 32

        self.classes = np.array(["square", "circle", "triangle"])
        self.labels = np.load(str(self.root_path / f"{split}_labels.npy"))
        self.ids = np.arange(len(self.labels))
        if selected_indices is not None:
            self.labels = self.labels[selected_indices]
            self.ids = self.ids[selected_indices]

        self.selected_indices = selected_indices

        self.mapping = mapping if mapping is not None else self.ids
        self.mapping = np.array(self.mapping)
        self.available_domains_mapping = domain_mapping
        if self.available_domains_mapping is None:
            domains = list(self.selected_domains)
            self.available_domains_mapping = [domains] * self.mapping.shape[0]

        if fetcher_params is None:
            fetcher_params = dict()
        for domain in self.available_domains.keys():
            if domain not in fetcher_params:
                fetcher_params[domain] = dict()

        self.data_fetchers = {
            domain: self.available_domains[domain](
                self.root_path, self.split, self.ids, self.labels, self.transforms, **fetcher_params[domain]
            )
            for domain in self.selected_domains
        }

        self.pre_saved_data = {}
        if pre_saved_latent_path is not None:
            self.use_pre_saved_latents(pre_saved_latent_path)

    def use_pre_saved_latents(self, pre_saved_latent_path: Optional[Dict[str, Any]]) -> None:
        if pre_saved_latent_path is not None:
            for domain_key in self.selected_domains:
                if domain_key in pre_saved_latent_path.keys():
                    self.pre_saved_data[domain_key] = load_pre_saved_latent(
                        self.root_path, self.split, pre_saved_latent_path, domain_key, self.ids
                    )
                    self.data_fetchers[domain_key] = PreSavedLatentDataFetcher(self.pre_saved_data[domain_key])

    def __len__(self) -> int:
        return self.mapping.shape[0]

    def __getitem__(self, item: int) -> SelectedDomainType:
        idx = self.mapping[item]
        mapping = self.available_domains_mapping[item]

        selected_domains = {}
        n_domains = 0

        for domain_key, fetcher in self.data_fetchers.items():
            if domain_key in mapping:
                fetched_items = fetcher.get_items(idx)
            else:
                fetched_items = fetcher.get_items(None)
            n_domains += fetched_items[0].item()
            selected_domains[domain_key] = fetched_items
        assert n_domains == len(mapping)
        if self.output_transform is not None:
            return self.output_transform(selected_domains)
        return selected_domains
