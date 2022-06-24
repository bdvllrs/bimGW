from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset

from bim_gw.datasets.simple_shapes.datasets import SimpleShapesDataset
from bim_gw.datasets.simple_shapes.utils import get_preprocess, create_ood_split, split_ood_sets, sample_domains
from bim_gw.utils.losses.compute_fid import compute_dataset_statistics


class SimpleShapesDataModule(LightningDataModule):
    def __init__(
            self, simple_shapes_folder, batch_size,
            num_workers=0, use_data_augmentation=False, prop_labelled_images =None,
            n_validation_domain_examples=None, split_ood=True,
            selected_domains=None,
            pre_saved_latent_paths=None,
            sync_uses_whole_dataset=False,
            with_actions=None
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
        self.with_actions = with_actions
        if with_actions is None:
            self.with_actions = 'a' in self.selected_domains.values()

        self.prop_labelled_images = prop_labelled_images

        self.num_channels = 3
        self.use_data_augmentation = use_data_augmentation

        ds = SimpleShapesDataset(simple_shapes_folder, "val", extend_dataset=False)
        self.classes = ds.classes
        self.val_dataset_size = len(ds)
        self.n_time_steps = 2
        self.is_setup = False

    def setup(self, stage=None):
        if not self.is_setup:
            val_transforms = {"v": get_preprocess()}
            train_transforms = {"v": get_preprocess(self.use_data_augmentation)}
            if stage == "fit" or stage is None:

                self.shapes_val = SimpleShapesDataset(self.simple_shapes_folder, "val",
                                                      transform=val_transforms,
                                                      selected_domains=self.selected_domains,
                                                      with_actions=self.with_actions)
                self.shapes_test = SimpleShapesDataset(self.simple_shapes_folder, "test",
                                                       transform=val_transforms,
                                                       selected_domains=self.selected_domains,
                                                       with_actions=self.with_actions)

                # train_set = SimpleShapesDataset(self.simple_shapes_folder, "train", extend_dataset=False, with_actions=self.with_actions)
                len_train_dataset = 1_000_000
                if self.sync_uses_whole_dataset:
                    sync_indices = np.arange(len_train_dataset)
                else:
                    sync_indices = np.arange(len_train_dataset // 2)
                train_set = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                                selected_indices=sync_indices,
                                                transform=train_transforms,
                                                selected_domains=self.selected_domains, with_actions=self.with_actions)

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

                self.shapes_val = split_ood_sets(self.shapes_val, id_ood_splits)
                self.shapes_test = split_ood_sets(self.shapes_test, id_ood_splits)
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

        self.is_setup = True

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
            "train": [{domain: [[] for _ in range(self.n_time_steps)] for domain in self.selected_domains.keys()}
                      for p in range(2)],
            "in_dist": [{domain: [[] for _ in range(self.n_time_steps)] for domain in self.selected_domains.keys()}
                        for p in range(2)],
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
                        if not self.with_actions:
                            example_item = [example_item]
                        for t in range(len(example_item)):
                            if not isinstance(example_item[t], tuple):
                                examples = []
                                for i in reconstruction_indices[used_dist]:
                                    example = used_set[i][p][domain]
                                    if self.with_actions:
                                        example = example[t]
                                    examples.append(example)
                                if isinstance(example_item[t], (int, float)):
                                    self.domain_examples[used_dist][p][domain][t] = torch.tensor(examples)
                                elif isinstance(example_item[t], torch.Tensor):
                                    self.domain_examples[used_dist][p][domain][t] = torch.stack(examples, dim=0)
                                else:
                                    self.domain_examples[used_dist][p][domain][t] = examples
                            else:
                                for k in range(len(example_item[t])):
                                    examples = []
                                    for i in reconstruction_indices[used_dist]:
                                        if self.with_actions:
                                            example = used_set[i][p][domain][t][k]
                                        else:
                                            example = used_set[i][p][domain][k]
                                        examples.append(example)
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
                        if not self.with_actions:
                            self.domain_examples[used_dist][p][domain] = self.domain_examples[used_dist][p][domain][0]

    def filter_sync_domains(self, train_set, allowed_indices):
        if self.prop_labelled_images < 1.:
            # Unlabel randomly some elements
            n_targets = len(train_set)
            np.random.shuffle(allowed_indices)
            num_labelled = int(self.prop_labelled_images * n_targets)
            labelled_elems = allowed_indices[:num_labelled]
            unlabelled_elems = allowed_indices[num_labelled:]
            print(f"Loaded {len(allowed_indices)} examples in train set.")
            train_set = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                            labelled_indices=labelled_elems,
                                            unlabelled_indices=unlabelled_elems,
                                            selected_domains=self.selected_domains,
                                            transform=train_set.transforms,
                                            output_transform=train_set.output_transform,
                                            with_actions=self.with_actions)
        return train_set

    def compute_inception_statistics(self, batch_size, device):
        train_ds = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                       transform={"v": get_preprocess(self.use_data_augmentation)},
                                       selected_domains={"v": "v"},
                                       output_transform=lambda d: d["v"][1])
        val_ds = SimpleShapesDataset(self.simple_shapes_folder, "val",
                                     transform={"v": get_preprocess(self.use_data_augmentation)},
                                     selected_domains={"v": "v"},
                                     output_transform=lambda d: d["v"][1])
        test_ds = SimpleShapesDataset(self.simple_shapes_folder, "test",
                                      transform={"v": get_preprocess(self.use_data_augmentation)},
                                      selected_domains={"v": "v"},
                                      output_transform=lambda d: d["v"][1])
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
