from pathlib import Path

import numpy as np

from bim_gw.datasets.simple_shapes.fetchers import VisualDataFetcher, AttributesDataFetcher, TextDataFetcher, \
    PreSavedLatentDataFetcher


class SimpleShapesDataset:
    available_domains = {
        "v": VisualDataFetcher,
        "attr": AttributesDataFetcher,
        "t": TextDataFetcher,
    }

    def __init__(self, path, split="train", selected_indices=None, min_dataset_size=None,
                 selected_domains=None, pre_saved_latent_path=None, transform=None, output_transform=None):
        """
        Args:
            path:
            split:
            transform:
            output_transform:
            selected_indices: To reduce the size of the dataset to given indices.
            min_dataset_size: Copies the data so that the effective size is the one given.
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

        self.classes = np.array(["square", "circle", "triangle"])
        self.labels = np.load(str(self.root_path / f"{split}_labels.npy"))

        self.ids = np.arange(self.labels.shape[0])

        if selected_indices is not None:
            self.labels = self.labels[selected_indices]
            self.ids = self.ids[selected_indices]

        if min_dataset_size is not None:
            original_size = len(self.labels)
            n_repeats = min_dataset_size // original_size + 1 * int(min_dataset_size % original_size > 0)
            self.ids = np.tile(self.ids, n_repeats)
            self.labels = np.tile(self.labels, (n_repeats, 1))

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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        selected_domains = {
            domain_key: fetcher[item] for domain_key, fetcher in self.data_fetchers.items()
        }

        if self.output_transform is not None:
            return self.output_transform(selected_domains)
        return selected_domains