import itertools
from pathlib import Path

import numpy as np

from bim_gw.datasets.simple_shapes.fetchers import VisualDataFetcher, AttributesDataFetcher, TextDataFetcher, \
    PreSavedLatentDataFetcher, ActionDataFetcher

class SimpleShapesDataset:
    available_domains = {
        "v": VisualDataFetcher,
        "attr": AttributesDataFetcher,
        "t": TextDataFetcher,
        "a": ActionDataFetcher,
    }

    def __init__(self, path, split="train", synced_domain_mapping=None, selected_indices=None,
                 selected_domains=None, pre_saved_latent_path=None, transform=None, output_transform=None,
                 extend_dataset=True):
        """
        Args:
            path:
            split:
            transform:
            output_transform:
            selected_indices:
            synced_domain_mapping: list with each available modality for each point
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
        self.with_actions = 'a' in self.selected_domains.values()

        self.classes = np.array(["square", "circle", "triangle"])
        self.synced_domain_mapping = synced_domain_mapping
        self.labels = np.load(str(self.root_path / f"{split}_labels.npy"))
        self.ids = np.arange(len(self.labels))
        if selected_indices is not None:
            self.labels = self.labels[selected_indices]
            self.ids = self.ids[selected_indices]

        domain_mapping_to_remove = []
        if self.synced_domain_mapping is not None:
            ids = []
            for k in range(len(self.synced_domain_mapping)):
                if (selected_indices is None or k in selected_indices) and len(self.synced_domain_mapping[k]):
                    ids.append(k)
                else:
                    domain_mapping_to_remove.append(k)
            self.ids = np.array(ids)
            self.labels = self.labels[self.ids]

        # Remove empty elements
        for idx in reversed(domain_mapping_to_remove):
            del self.synced_domain_mapping[idx]

        self.input_domains = self.synced_domain_mapping
        self.target_domains = self.synced_domain_mapping

        if extend_dataset:
            pass
            # self.define_targets()

        self.all_fetchers = {
            name: fetcher(self.root_path, self.split, self.ids, self.labels, self.transforms) for name, fetcher in
            self.available_domains.items()
        }

        self.data_fetchers = {
            domain_key: self.all_fetchers[domain]
            for domain_key, domain in self.selected_domains.items()
        }

        if pre_saved_latent_path is not None:
            self.use_pre_saved_latents(pre_saved_latent_path)

    def use_pre_saved_latents(self, pre_saved_latent_path):
        if pre_saved_latent_path is not None:
            for key, path in pre_saved_latent_path.items():
                if key in self.data_fetchers.keys():
                    self.data_fetchers[key] = PreSavedLatentDataFetcher(
                        self.root_path / "saved_latents" / self.split / path, self.ids)

    def define_targets(self):
        labels = []
        ids = []
        targets = []
        for k in range(self.labels.shape[0]):
            domains = list(self.selected_domains.values())
            if self.synced_domain_mapping is not None:
                domains = self.synced_domain_mapping[k]
            for r in range(1, len(domains) + 1):
                combinations = itertools.combinations(domains, r)
                for combination in combinations:
                    labels.append(combination)
                    targets.append(domains)
                    ids.append(self.ids[k])
        self.input_domains = labels
        self.target_domains = targets
        if self.synced_domain_mapping is not None:
            self.synced_domain_mapping = targets
        self.ids = np.array(ids)
        self.labels = self.labels[self.ids]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        items = []
        for mapping in [self.input_domains, self.target_domains]:
            selected_domains = {}
            for domain_key, fetcher in self.data_fetchers.items():
                time_steps = []
                if mapping is None or domain_key in mapping[item]:
                    time_steps.append(0)
                if self.with_actions and (mapping is None or domain_key + "_f" in mapping[item]):
                    time_steps.append(1)
                fetched_items =  fetcher.get_items(item, time_steps)
                if not self.with_actions:
                    fetched_items = fetched_items[0]
                selected_domains[domain_key] = fetched_items

            if self.output_transform is not None:
                return self.output_transform(selected_domains)
            items.append(selected_domains)
        return items
