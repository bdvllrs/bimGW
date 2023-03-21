import logging
from typing import List, Optional, Set, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule


def split_indices_prop(
    all_indices,
    allowed_indices: Set[int],
    prop: float
) -> Tuple[List[int], List[int]]:
    # Unlabel randomly some elements
    n_targets = len(allowed_indices)
    all_indices = np.random.permutation(all_indices)
    allowed_indices = [i for i in all_indices if i in allowed_indices]
    num_labelled = int(prop * n_targets)
    selected = allowed_indices[:num_labelled]
    rest = allowed_indices[num_labelled:]
    return selected, rest


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
                    domain: [[] for _ in range(self.n_time_steps)] for domain
                    in
                    self.selected_domains}

        # add t examples
        for set_name, used_set in [("train", {"in_dist": train_set}),
                                   ("val", val_set), ("test", test_set)]:
            for used_dist in range(2):
                used_dist_name = "in_dist" if used_dist == 0 else "ood"
                dist_indices = reconstruction_indices[set_name][used_dist]
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
                                self.domain_examples[set_name][used_dist][
                                    domain] = torch.tensor(examples)
                            elif isinstance(example_item, torch.Tensor):
                                self.domain_examples[set_name][used_dist][
                                    domain] = torch.stack(examples, dim=0)
                            else:
                                self.domain_examples[set_name][used_dist][
                                    domain] = examples
                        else:
                            for k in range(len(example_item)):
                                examples = []
                                for i in dist_indices:
                                    example = cur_set[i][domain][k]
                                    examples.append(example)
                                if isinstance(example_item[k], (int, float)):
                                    self.domain_examples[set_name][used_dist][
                                        domain].append(
                                        torch.tensor(examples)
                                    )
                                elif isinstance(example_item[k], torch.Tensor):
                                    self.domain_examples[set_name][used_dist][
                                        domain].append(
                                        torch.stack(examples, dim=0)
                                    )
                                else:
                                    self.domain_examples[set_name][used_dist][
                                        domain].append(examples)
                            self.domain_examples[set_name][used_dist][
                                domain] = tuple(
                                self.domain_examples[set_name][used_dist][
                                    domain]
                            )

    def filter_sync_domains(
        self, n_total_examples: int, allowed_indices: Set[int]
    ) -> Tuple[List[int], List[List[str]]]:
        all_indices = np.arange(n_total_examples)
        # Only keep proportion of images in self.prop_available_images.
        # This split is done regardless of the ood split.
        allowed_indices, _ = split_indices_prop(
            all_indices,
            allowed_indices, self.prop_available_images
        )
        logging.debug(f"Loaded {len(allowed_indices)} examples in train set.")

        prop_2_domains = self.prop_labelled_images / self.prop_available_images
        # prop_3_domains = self.prop_labelled_images[1]
        # assert prop_3_domains <= prop_2_domains, "Must have less
        # synchronization with 3 than 2 domains"
        mapping = None
        domain_mapping = None
        if prop_2_domains < 1:
            domains = list(self.selected_domains)
            original_size = len(allowed_indices)
            labelled_size = int(original_size * prop_2_domains)
            n_repeats = ((len(domains) * original_size) // labelled_size +
                         int(original_size % labelled_size > 0))
            mapping = []
            domain_mapping = []

            # labelled_elems, rest_elems = split_indices_prop(
            # allowed_indices, prop_3_domains)

            done = []
            if self.remove_sync_domains is not None:
                done = self.remove_sync_domains[:]
            # Add sync domains
            for domain_1 in domains:
                mapping.extend(allowed_indices[:])
                domain_mapping.extend([[domain_1]] * original_size)

                for domain_2 in domains:
                    if domain_1 != domain_2 and (
                            domain_2, domain_1) not in done and (
                            domain_1, domain_2) not in done:
                        done.append((domain_1, domain_2))
                        domain_items, _ = split_indices_prop(
                            all_indices,
                            set(allowed_indices), prop_2_domains
                        )
                        domain_items = np.tile(domain_items, n_repeats)
                        mapping.extend(domain_items)
                        domain_mapping.extend(
                            [[domain_1, domain_2]] * len(domain_items)
                        )

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
