import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule

from bim_gw.datasets.simple_shapes.datasets import AVAILABLE_DOMAINS


class DataModule(LightningDataModule):
    def __init__(
        self, batch_size: int,
        num_workers: int = 0, prop_labelled_images: float = 1.,
        prop_available_images: float = 1.,
        removed_sync_domains: Optional[List[List[str]]] = None,
        n_validation_domain_examples: int = 32, split_ood: bool = True,
        selected_domains: Optional[List[str]] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ood = split_ood
        self.n_domain_examples = batch_size
        if n_validation_domain_examples is not None:
            self.n_domain_examples = n_validation_domain_examples
        self.domain_examples = None
        self.ood_boundaries = None
        self.selected_domains = selected_domains
        if self.selected_domains is None:
            self.selected_domains = list(AVAILABLE_DOMAINS.keys())

        self.prop_labelled_images = prop_labelled_images
        self.prop_available_images = prop_available_images
        if self.prop_available_images < self.prop_labelled_images:
            raise ValueError(
                "prop_available_images must be >= prop_labelled_images"
            )

        # Remove sync for some combination of domains
        self.remove_sync_domains = removed_sync_domains

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def set_validation_examples(self, train_set, val_set, test_set):
        reconstruction_indices = {
            "train": [
                torch.randint(len(train_set), size=(self.n_domain_examples,)),
                None],
            "val": [torch.randint(
                len(val_set["in_dist"]), size=(self.n_domain_examples,)
            ), None],
            "test": [torch.randint(
                len(test_set["in_dist"]), size=(self.n_domain_examples,)
            ), None]
        }

        if val_set["ood"] is not None:
            reconstruction_indices["val"][1] = torch.randint(
                len(val_set["ood"]),
                size=(self.n_domain_examples,)
            )
        if test_set["ood"] is not None:
            reconstruction_indices["test"][1] = torch.randint(
                len(test_set["ood"]),
                size=(self.n_domain_examples,)
            )

        self.domain_examples = {
            "train": [{domain: [] for domain in self.selected_domains}, None],
            "val": [{domain: [] for domain in self.selected_domains}, None],
            "test": [{domain: [] for domain in self.selected_domains}, None],
        }

        if self.split_ood:
            for set_name in ["val", "test"]:
                self.domain_examples[set_name][1] = {
                    domain: [[] for _ in range(self.n_time_steps)]
                    for domain in self.selected_domains
                }

        all_sets = [
            ("train", {"in_dist": train_set}),
            ("val", val_set),
            ("test", test_set)
        ]

        for set_name, used_set in all_sets:
            for used_dist in range(2):
                used_dist_name = "in_dist" if used_dist == 0 else "ood"
                dist_indices = reconstruction_indices[set_name][used_dist]
                dist_examples = self.domain_examples[set_name][used_dist]
                if dist_indices is not None:
                    cur_set = used_set[used_dist_name]
                    for domain in self.selected_domains:
                        example_item = cur_set[0][domain]
                        if not isinstance(example_item, tuple):
                            examples = []
                            for i in dist_indices:
                                example = cur_set[i][domain]
                                examples.append(example)
                            if isinstance(example_item, (int, float)):
                                dist_examples[domain] = torch.tensor(examples)
                            elif isinstance(example_item, torch.Tensor):
                                dist_examples[domain] = torch.stack(
                                    examples, dim=0
                                )
                            else:
                                dist_examples[domain] = examples
                        else:
                            for k in range(len(example_item)):
                                examples = []
                                for i in dist_indices:
                                    example = cur_set[i][domain][k]
                                    examples.append(example)
                                if isinstance(example_item[k], (int, float)):
                                    dist_examples[domain].append(
                                        torch.tensor(examples)
                                    )
                                elif isinstance(example_item[k], torch.Tensor):
                                    dist_examples[domain].append(
                                        torch.stack(examples, dim=0)
                                    )
                                else:
                                    dist_examples[domain].append(examples)
                            dist_examples[domain] = tuple(
                                dist_examples[domain]
                            )

    def filter_sync_domains(
        self, allowed_indices: List[int]
    ) -> Tuple[List[int], List[List[str]]]:
        domains = list(self.selected_domains)
        # permute for number of couples of domains
        permuted_indices = np.random.permutation(allowed_indices)
        logging.debug(f"Loaded {len(allowed_indices)} examples in train set.")

        prop_2_domains = self.prop_labelled_images / self.prop_available_images
        original_size = int(
            len(allowed_indices) * self.prop_available_images
        )

        if prop_2_domains == 1 and self.prop_available_images == 1:
            return None, None

        sync_split = int(prop_2_domains * original_size)
        sync_items = permuted_indices[:sync_split]
        rest = permuted_indices[sync_split:]

        mapping = []
        domain_mapping = []
        if prop_2_domains < 1:
            labelled_size = int(original_size * prop_2_domains)
            n_repeats = ((len(domains) * original_size) // labelled_size +
                         int(original_size % labelled_size > 0))

            domain_items = np.tile(sync_items, n_repeats)
            mapping.extend(domain_items)
            domain_mapping.extend(
                [domains] * len(domain_items)
            )

        unsync_domain_items = permuted_indices
        if self.prop_available_images < 1:
            n_unsync = int(
                self.prop_available_images * len(allowed_indices)
            ) - sync_split
            unsync_items = rest[:n_unsync]
            unsync_domain_items = np.concatenate((unsync_items, sync_items))
        mapping.extend(unsync_domain_items)
        domain_mapping.extend([[domains[0]]] * len(unsync_domain_items))
        mapping.extend(unsync_domain_items)
        domain_mapping.extend([[domains[1]]] * len(unsync_domain_items))

        return mapping, domain_mapping

    def train_dataloader(
        self, shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_set,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self) -> List[torch.utils.data.DataLoader]:
        dataloaders = [
            torch.utils.data.DataLoader(
                self.val_set["in_dist"], self.batch_size,
                num_workers=self.num_workers, pin_memory=True
            ),
        ]
        if self.val_set["ood"] is not None:
            dataloaders.append(
                torch.utils.data.DataLoader(
                    self.val_set["ood"], self.batch_size,
                    num_workers=self.num_workers, pin_memory=True
                )
            )
        return dataloaders

    def test_dataloader(self) -> List[torch.utils.data.DataLoader]:
        dataloaders = [
            torch.utils.data.DataLoader(
                self.test_set["in_dist"], self.batch_size,
                num_workers=self.num_workers, pin_memory=True
            ),
        ]

        if self.test_set["ood"] is not None:
            dataloaders.append(
                torch.utils.data.DataLoader(
                    self.test_set["ood"], self.batch_size,
                    num_workers=self.num_workers, pin_memory=True
                )
            )
        return dataloaders
