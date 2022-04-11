from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset

from bim_gw.datasets.simple_shapes.datasets import SimpleShapesDataset
from bim_gw.datasets.simple_shapes.utils import get_preprocess, create_ood_split, split_odd_sets, sample_domains
from bim_gw.utils.losses.compute_fid import compute_dataset_statistics


class SimpleShapesDataModule(LightningDataModule):
    def __init__(
            self, simple_shapes_folder, batch_size,
            num_workers=0, use_data_augmentation=False, prop_sync_domains=None,
            n_validation_domain_examples=None, split_ood=True,
            selected_domains=None,
            pre_saved_latent_paths=None,
            sync_uses_whole_dataset=False
    ):
        super().__init__()
        self.simple_shapes_folder = Path(simple_shapes_folder)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = 32
        self.split_ood = split_ood
        self.domain_examples = n_validation_domain_examples if n_validation_domain_examples is not None else batch_size
        self.ood_boundaries = None
        self.selected_domains = selected_domains
        self.pre_saved_latent_paths = pre_saved_latent_paths
        self.sync_uses_whole_dataset = sync_uses_whole_dataset

        self.prop_sync_domains = prop_sync_domains

        self.num_channels = 3
        self.use_data_augmentation = use_data_augmentation

        ds = SimpleShapesDataset(simple_shapes_folder, "val", extend_dataset=False)
        self.classes = ds.classes
        self.val_dataset_size = len(ds)
        self.n_time_steps = 2

    def setup(self, stage=None):
        val_transforms = {"v": get_preprocess()}
        train_transforms = {"v": get_preprocess(self.use_data_augmentation)}
        if stage == "fit" or stage is None:

            self.shapes_val = SimpleShapesDataset(self.simple_shapes_folder, "val",
                                                  transform=val_transforms,
                                                  selected_domains=self.selected_domains)
            self.shapes_test = SimpleShapesDataset(self.simple_shapes_folder, "test",
                                                   transform=val_transforms,
                                                   selected_domains=self.selected_domains)

            train_set = SimpleShapesDataset(self.simple_shapes_folder, "train", extend_dataset=False)
            len_train_dataset = len(train_set)
            if self.sync_uses_whole_dataset:
                sync_indices = np.arange(len_train_dataset)
            else:
                sync_indices = np.arange(len_train_dataset // 2)
            train_set = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                            selected_indices=sync_indices,
                                            transform=train_transforms,
                                            selected_domains=self.selected_domains)

            if self.split_ood:
                id_ood_splits, ood_boundaries = create_ood_split(
                    [train_set, self.shapes_val, self.shapes_test])
                self.ood_boundaries = ood_boundaries

                target_indices = np.unique(id_ood_splits[0][0])

                print("Val set in dist size", len(id_ood_splits[1][0]))
                print("Val set OOD size", len(id_ood_splits[1][1]))
                print("Test set in dist size", len(id_ood_splits[2][0]))
                print("Test set OOD size", len(id_ood_splits[2][1]))
            else:
                id_ood_splits = None
                target_indices = train_set.ids

            self.shapes_val = split_odd_sets(self.shapes_val, id_ood_splits)
            self.shapes_test = split_odd_sets(self.shapes_test, id_ood_splits)
            self.shapes_train = self.filter_sync_domains(train_set, target_indices)

        self.set_validation_examples(
            self.shapes_val if stage == "fit" else self.shapes_test,
            self.shapes_train
        )

        # Use pre saved latents if provided.
        for shapes_set in [{"train": self.shapes_train}, self.shapes_val, self.shapes_test]:
            for dataset in shapes_set.values():
                if dataset is not None:
                    if isinstance(dataset, Subset):
                        dataset = dataset.dataset
                    dataset.use_pre_saved_latents(self.pre_saved_latent_paths)

    def set_validation_examples(self, test_set, train_set):
        reconstruction_indices = {
            "train": torch.randint(len(train_set), size=(self.domain_examples,)),
            "in_dist": torch.randint(len(test_set["in_dist"]), size=(self.domain_examples,)),
            "ood": None,
        }

        if test_set["ood"] is not None:
            reconstruction_indices["ood"] = torch.randint(len(test_set["ood"]),
                                                          size=(self.domain_examples,))

        self.domain_examples = {
            "train": [{domain: [[] for _ in range(self.n_time_steps)] for domain in self.selected_domains.keys()} for p
                      in range(2)],
            "in_dist": [{domain: [[] for _ in range(self.n_time_steps)] for domain in self.selected_domains.keys()} for
                        p
                        in range(2)],
            "ood": None,
        }

        used_dists = ["train", "in_dist"]

        if self.split_ood:
            self.domain_examples["ood"] = [
                {domain: [[] for _ in range(self.n_time_steps)] for domain in
                 self.selected_domains.keys()} for p in range(2)
            ]
            used_dists.append("ood")

        # add t examples
        for used_dist in used_dists:
            used_set = train_set if used_dist == "train" else test_set[used_dist]
            if reconstruction_indices[used_dist] is not None:
                for p in range(2):  # input or target
                    for domain in self.selected_domains.keys():
                        example_item = used_set[0][p][domain]
                        for t in range(len(example_item)):
                            if not isinstance(example_item[t], tuple):
                                examples = [used_set[i][p][domain][t] for i in
                                            reconstruction_indices[used_dist]]
                                if isinstance(example_item[t], (int, float)):
                                    self.domain_examples[used_dist][p][domain][t] = torch.tensor(examples)
                                elif isinstance(example_item[t], torch.Tensor):
                                    self.domain_examples[used_dist][p][domain][t] = torch.stack(examples, dim=0)
                                else:
                                    self.domain_examples[used_dist][p][domain][t] = examples
                            else:
                                for k in range(len(example_item[t])):
                                    examples = [used_set[i][p][domain][t][k] for i in
                                                reconstruction_indices[used_dist]]
                                    if isinstance(example_item[t][k], (int, float)):
                                        self.domain_examples[used_dist][p][domain][t].append(
                                            torch.tensor(examples)
                                        )
                                    elif isinstance(example_item[t][k], torch.Tensor):
                                        self.domain_examples[used_dist][p][domain][t].append(
                                            torch.stack(examples, dim=0)
                                        )
                                    else:
                                        self.domain_examples[used_dist][p][domain][t].append(examples)
                                self.domain_examples[used_dist][p][domain][t] = tuple(
                                    self.domain_examples[used_dist][p][domain][t])

    def filter_sync_domains(self, train_set, allowed_indices):
        # Unlabel randomly some elements
        n_targets = len(allowed_indices)
        np.random.shuffle(allowed_indices)
        sync_domain_mapping = {}
        sampler_domain_map = {}
        available_domains = {}
        for domain_name, domain in train_set.selected_domains.items():
            available_domains[domain_name] = domain
            if domain != "a":
                available_domains[domain_name + "_f"] = domain + "_f"
        if self.prop_sync_domains is not None:
            last_idx = 0
            for key, prop in self.prop_sync_domains.items():
                n_items = int(prop * n_targets)
                sampled_domains = sample_domains(available_domains, key, n_items)
                for k, idx in enumerate(allowed_indices[last_idx:last_idx + n_items]):
                    domain_key = str(sorted(sampled_domains[k]))
                    if domain_key not in sampler_domain_map.keys():
                        sampler_domain_map[domain_key] = []
                    sync_domain_mapping[idx] = list(sampled_domains[k])
                    sampler_domain_map[domain_key].append(idx)

                last_idx += n_items

        domain_mapping = []
        for k in range(len(train_set)):
            if k in sync_domain_mapping.keys():
                domain_mapping.append(sync_domain_mapping[k])
            else:
                domain_mapping.append([])

        print(f"Training using {len(allowed_indices)} examples.")
        train_set = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                        domain_mapping,
                                        selected_domains=self.selected_domains,
                                        transform=train_set.transforms,
                                        output_transform=train_set.output_transform)
        return train_set

    def compute_inception_statistics(self, batch_size, device):
        train_ds = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                       transform={"v": get_preprocess(self.use_data_augmentation)},
                                       selected_domains={"v": "v"},
                                       output_transform=lambda d: (d["v"], 0))
        val_ds = SimpleShapesDataset(self.simple_shapes_folder, "val",
                                     transform={"v": get_preprocess(self.use_data_augmentation)},
                                     selected_domains={"v": "v"},
                                     output_transform=lambda d: (d["v"], 0))
        test_ds = SimpleShapesDataset(self.simple_shapes_folder, "test",
                                      transform={"v": get_preprocess(self.use_data_augmentation)},
                                      selected_domains={"v": "v"},
                                      output_transform=lambda d: (d["v"], 0))
        self.inception_stats_path_train = compute_dataset_statistics(train_ds, self.simple_shapes_folder,
                                                                     "shapes_train",
                                                                     batch_size, device)
        self.inception_stats_path_val = compute_dataset_statistics(val_ds, self.simple_shapes_folder, "shapes_val",
                                                                   batch_size, device)

        self.inception_stats_path_test = compute_dataset_statistics(test_ds, self.simple_shapes_folder, "shapes_test",
                                                                    batch_size, device)

    def train_dataloader(self, shuffle=True):
        return torch.utils.data.DataLoader(self.shapes_train,
                                           shuffle=shuffle,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=True)

    def val_dataloader(self):
        dataloaders = [
            torch.utils.data.DataLoader(self.shapes_val["in_dist"], self.batch_size,
                                        num_workers=self.num_workers, pin_memory=True),
        ]
        if self.shapes_val["ood"] is not None:
            dataloaders.append(
                torch.utils.data.DataLoader(self.shapes_val["ood"], self.batch_size,
                                            num_workers=self.num_workers, pin_memory=True)
            )
        return dataloaders

    def test_dataloader(self):
        dataloaders = [
            torch.utils.data.DataLoader(self.shapes_test["in_dist"], self.batch_size,
                                        num_workers=self.num_workers, pin_memory=True),
        ]

        if self.shapes_test["ood"] is not None:
            dataloaders.append(
                torch.utils.data.DataLoader(self.shapes_test["ood"], self.batch_size,
                                            num_workers=self.num_workers, pin_memory=True)
            )
        return dataloaders