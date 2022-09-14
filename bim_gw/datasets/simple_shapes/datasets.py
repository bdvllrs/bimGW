import itertools
import math
from pathlib import Path

import numpy as np

from bim_gw.datasets.simple_shapes.fetchers import VisualDataFetcher, AttributesDataFetcher, TextDataFetcher, \
    PreSavedLatentDataFetcher, ActionDataFetcher
from bim_gw.datasets.pre_saved_latents import load_pre_saved_latent


class SimpleShapesDataset:
    available_domains = {
        "v": VisualDataFetcher,
        "attr": AttributesDataFetcher,
        "t": TextDataFetcher,
        "a": ActionDataFetcher
    }

    def __init__(self, path, split="train", mapping=None, domain_mapping=None, selected_indices=None,
                 selected_domains=None, pre_saved_latent_path=None, transform=None, output_transform=None,
                 add_unimodal=True, with_actions=None, fetcher_params=None):
        """
        Args:
            path:
            split:
            transform:
            output_transform:
            selected_indices:
        """
        assert split in ["train", "val", "test"]
        self.selected_domains = {domain: domain for domain in
                                 self.available_domains.keys()} if selected_domains is None else selected_domains
        self.root_path = Path(path)
        self.transforms = {domain: (transform[domain] if (transform is not None and domain in transform) else None)
                           for domain in self.available_domains.keys()}
        self.output_transform = output_transform
        self.split = split
        self.img_size = 32
        self.with_actions = with_actions
        self.add_unimodal = add_unimodal
        if with_actions is None:
            self.with_actions = 'a' in self.selected_domains.values()

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
            domains = list(self.selected_domains.keys())
            self.available_domains_mapping = [[domains]] * self.mapping.shape[0]

        if fetcher_params is None:
            fetcher_params = dict()
        for domain in self.available_domains.keys():
            if domain not in fetcher_params:
                fetcher_params[domain] = dict()

        self.data_fetchers = {
            domain_key: self.available_domains[domain](
                self.root_path, self.split, self.ids, self.labels, self.transforms, **fetcher_params[domain])
            for domain_key, domain in self.selected_domains.items()
        }

        self.pre_saved_data = {}
        if pre_saved_latent_path is not None:
            self.use_pre_saved_latents(pre_saved_latent_path)

    def use_pre_saved_latents(self, pre_saved_latent_path):
        if pre_saved_latent_path is not None:
            for key, domain_key in self.selected_domains.items():
                if domain_key in pre_saved_latent_path.keys():
                    self.pre_saved_data[domain_key] = load_pre_saved_latent(
                        self.root_path, self.split, pre_saved_latent_path, domain_key, self.ids)
                    self.data_fetchers[key] = PreSavedLatentDataFetcher(self.pre_saved_data[domain_key])

    def __len__(self):
        return self.mapping.shape[0]

    def __getitem__(self, item):
        idx = self.mapping[item]
        domains = self.available_domains_mapping[item]

        items = []
        for mapping in [domains, domains]:
            selected_domains = {}
            n_domains = 0

            for domain_key, fetcher in self.data_fetchers.items():
                time_steps = []
                if domain_key in mapping:
                    time_steps.append(0)
                if self.with_actions and (domain_key + "_f" in mapping):
                    time_steps.append(1)
                fetched_items = fetcher.get_items(idx, time_steps)
                n_domains += fetched_items[0][0].item()
                if not self.with_actions:
                    fetched_items = fetched_items[0]
                selected_domains[domain_key] = fetched_items
            if n_domains == len(self.data_fetchers.keys()):
                assert idx in self.labelled_indices
            if self.output_transform is not None:
                return self.output_transform(selected_domains)
            items.append(selected_domains)
        return items
