from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset

from bim_gw.datasets.simple_shapes.datasets import SimpleShapesDataset
from bim_gw.datasets.simple_shapes.utils import get_preprocess, create_ood_split, split_ood_sets
from bim_gw.modules.domain_modules import VAE
from bim_gw.modules.domain_modules.simple_shapes import SimpleShapesAttributes, SimpleShapesText
from bim_gw.utils.registers import DomainRegister
from bim_gw.utils.losses.compute_fid import compute_dataset_statistics


def add_domains_to_register():
    domain_register = DomainRegister()
    domain_register.add("v", lambda args, img_size=None: VAE.load_from_checkpoint(
        args.global_workspace.vae_checkpoint,
        mmd_loss_coef=args.global_workspace.vae_mmd_loss_coef,
        kl_loss_coef=args.global_workspace.vae_kl_loss_coef,
        strict=False
    ))
    domain_register.add("attr", lambda args, img_size: SimpleShapesAttributes(img_size))
    domain_register.add("t", lambda args, img_size=None: SimpleShapesText.load_from_checkpoint(
        args.global_workspace.lm_checkpoint,
        bert_path=args.global_workspace.bert_path,
        z_size=args.lm.z_size,
        hidden_size=args.lm.hidden_size
    ))


def split_indices_prop(allowed_indices, prop):
    # Unlabel randomly some elements
    n_targets = len(allowed_indices)
    assert np.unique(allowed_indices, return_counts=True)[1].max() == 1
    np.random.shuffle(allowed_indices)
    num_labelled = int(prop * n_targets)
    selected = allowed_indices[:num_labelled]
    rest = allowed_indices[num_labelled:]
    assert len(selected) + len(rest) == len(allowed_indices)
    assert len(selected) / len(allowed_indices) == prop
    assert np.intersect1d(selected, rest).shape[0] == 0
    return selected, rest


class SimpleShapesDataModule(LightningDataModule):
    def __init__(
            self, simple_shapes_folder, batch_size,
            num_workers=0, prop_labelled_images=None,
            removed_sync_domains=None,
            n_validation_domain_examples=None, split_ood=True,
            selected_domains=None,
            pre_saved_latent_paths=None,
            sync_uses_whole_dataset=False,
            add_unimodal=True,
            fetcher_params=None
    ):
        super().__init__()
        self.simple_shapes_folder = Path(simple_shapes_folder)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = 32
        self.split_ood = split_ood
        self.n_domain_examples = n_validation_domain_examples if n_validation_domain_examples is not None else batch_size
        self.domain_examples = None
        self.ood_boundaries = None
        self.selected_domains = selected_domains
        self.pre_saved_latent_paths = pre_saved_latent_paths
        self.sync_uses_whole_dataset = sync_uses_whole_dataset
        self.add_unimodal = add_unimodal
        self.fetcher_params = fetcher_params

        self.prop_labelled_images = prop_labelled_images
        # Remove sync for some combination of domains
        self.remove_sync_domains = removed_sync_domains

        self.num_channels = 3

        ds = SimpleShapesDataset(simple_shapes_folder, "val", selected_domains=self.selected_domains,
                                 fetcher_params=self.fetcher_params)
        self.classes = ds.classes
        self.val_dataset_size = len(ds)
        self.is_setup = False

        add_domains_to_register()

    def setup(self, stage=None):
        if not self.is_setup:
            val_transforms = {"v": get_preprocess()}
            train_transforms = {"v": get_preprocess()}
            if stage == "fit" or stage is None:
                self.shapes_val = SimpleShapesDataset(self.simple_shapes_folder, "val",
                                                      transform=val_transforms,
                                                      selected_domains=self.selected_domains,
                                                      fetcher_params=self.fetcher_params)
                self.shapes_test = SimpleShapesDataset(self.simple_shapes_folder, "test",
                                                       transform=val_transforms,
                                                       selected_domains=self.selected_domains,
                                                       fetcher_params=self.fetcher_params)

                # train_set = SimpleShapesDataset(self.simple_shapes_folder, "train", extend_dataset=False, with_actions=self.with_actions)
                len_train_dataset = 1_000_000
                if self.sync_uses_whole_dataset:
                    sync_indices = np.arange(len_train_dataset)
                else:
                    sync_indices = np.arange(len_train_dataset // 2)
                train_set = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                                selected_indices=sync_indices,
                                                transform=train_transforms,
                                                selected_domains=self.selected_domains,
                                                fetcher_params=self.fetcher_params)

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

                if self.add_unimodal:
                    self.shapes_train = self.filter_sync_domains(train_set, target_indices)
                else:
                    self.shapes_train = train_set

            self.set_validation_examples(
                self.shapes_train,
                self.shapes_val,
                self.shapes_test
            )

            # Use pre saved latents if provided.
            for shapes_set in [{"train": self.shapes_train}, self.shapes_val, self.shapes_test]:
                for dataset in shapes_set.values():
                    if dataset is not None:
                        if isinstance(dataset, Subset):
                            dataset = dataset.dataset
                        dataset.use_pre_saved_latents(self.pre_saved_latent_paths)

        self.is_setup = True

    def set_validation_examples(self, train_set, val_set, test_set):
        reconstruction_indices = {
            "train": [torch.randint(len(train_set), size=(self.n_domain_examples,)), None],
            "val": [torch.randint(len(val_set["in_dist"]), size=(self.n_domain_examples,)), None],
            "test": [torch.randint(len(test_set["in_dist"]), size=(self.n_domain_examples,)), None]
        }

        if val_set["ood"] is not None:
            reconstruction_indices["val"][1] = torch.randint(len(val_set["ood"]),
                                                             size=(self.n_domain_examples,))
        if test_set["ood"] is not None:
            reconstruction_indices["test"][1] = torch.randint(len(test_set["ood"]),
                                                              size=(self.n_domain_examples,))

        self.domain_examples = {
            "train": [{domain: [] for domain in self.selected_domains}, None],
            "val": [{domain: [] for domain in self.selected_domains}, None],
            "test": [{domain: [] for domain in self.selected_domains}, None],
        }

        if self.split_ood:
            for set_name in ["val", "test"]:
                self.domain_examples[set_name][1] = {domain: [[] for _ in range(self.n_time_steps)] for domain in
                                                     self.selected_domains}

        # add t examples
        for set_name, used_set in [("train", {"in_dist": train_set}), ("val", val_set), ("test", test_set)]:
            for used_dist in range(2):
                used_dist_name = "in_dist" if used_dist == 0 else "ood"
                if reconstruction_indices[set_name][used_dist] is not None:
                    for domain in self.selected_domains:
                        example_item = used_set[used_dist_name][0][domain]
                        if not isinstance(example_item, tuple):
                            examples = []
                            for i in reconstruction_indices[set_name][used_dist]:
                                example = used_set[used_dist_name][i][domain]
                                examples.append(example)
                            if isinstance(example_item, (int, float)):
                                self.domain_examples[set_name][used_dist][domain] = torch.tensor(examples)
                            elif isinstance(example_item, torch.Tensor):
                                self.domain_examples[set_name][used_dist][domain] = torch.stack(examples, dim=0)
                            else:
                                self.domain_examples[set_name][used_dist][domain] = examples
                        else:
                            for k in range(len(example_item)):
                                examples = []
                                for i in reconstruction_indices[set_name][used_dist]:
                                    example = used_set[used_dist_name][i][domain][k]
                                    examples.append(example)
                                if isinstance(example_item[k], (int, float)):
                                    self.domain_examples[set_name][used_dist][domain].append(
                                        torch.tensor(examples)
                                    )
                                elif isinstance(example_item[k], torch.Tensor):
                                    self.domain_examples[set_name][used_dist][domain].append(
                                        torch.stack(examples, dim=0)
                                    )
                                else:
                                    self.domain_examples[set_name][used_dist][domain].append(examples)
                            self.domain_examples[set_name][used_dist][domain] = tuple(
                                self.domain_examples[set_name][used_dist][domain])

    def filter_sync_domains(self, train_set, allowed_indices):
        prop_2_domains = self.prop_labelled_images
        # prop_3_domains = self.prop_labelled_images[1]
        # assert prop_3_domains <= prop_2_domains, "Must have less synchronization with 3 than 2 domains"
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

            # labelled_elems, rest_elems = split_indices_prop(allowed_indices, prop_3_domains)

            done = [] if self.remove_sync_domains is None else self.remove_sync_domains[:]
            # Add sync domains
            for domain_1 in domains:
                mapping.extend(allowed_indices[:])
                domain_mapping.extend([[domain_1]] * original_size)

                for domain_2 in domains:
                    if domain_1 != domain_2 and (domain_2, domain_1) not in done and (domain_1, domain_2) not in done:
                        done.append((domain_1, domain_2))
                        domain_items, _ = split_indices_prop(allowed_indices, prop_2_domains)
                        domain_items = np.tile(domain_items, n_repeats)
                        mapping.extend(domain_items)
                        domain_mapping.extend([[domain_1, domain_2]] * len(domain_items))

        print(f"Loaded {len(allowed_indices)} examples in train set.")
        train_set = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                        mapping=mapping,
                                        domain_mapping=domain_mapping,
                                        selected_domains=self.selected_domains,
                                        transform=train_set.transforms,
                                        output_transform=train_set.output_transform,
                                        fetcher_params=self.fetcher_params)
        return train_set

    def compute_inception_statistics(self, batch_size, device):
        train_ds = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                       transform={"v": get_preprocess()},
                                       selected_domains=["v"],
                                       output_transform=lambda d: d["v"][1],
                                       fetcher_params=self.fetcher_params)
        val_ds = SimpleShapesDataset(self.simple_shapes_folder, "val",
                                     transform={"v": get_preprocess()},
                                     selected_domains=["v"],
                                     output_transform=lambda d: d["v"][1],
                                     fetcher_params=self.fetcher_params)
        test_ds = SimpleShapesDataset(self.simple_shapes_folder, "test",
                                      transform={"v": get_preprocess()},
                                      selected_domains=["v"],
                                      output_transform=lambda d: d["v"][1],
                                      fetcher_params=self.fetcher_params)
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
