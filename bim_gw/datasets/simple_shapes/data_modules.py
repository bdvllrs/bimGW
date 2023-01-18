from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset

from bim_gw.datasets.data_module import DataModule
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


class SimpleShapesDataModule(DataModule):
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
        super().__init__(batch_size, num_workers, prop_labelled_images, removed_sync_domains,
                         n_validation_domain_examples, split_ood, selected_domains, pre_saved_latent_paths,
                         add_unimodal, fetcher_params)

        self.simple_shapes_folder = Path(simple_shapes_folder)
        self.img_size = 32
        self.sync_uses_whole_dataset = sync_uses_whole_dataset
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
                self.val_set = SimpleShapesDataset(self.simple_shapes_folder, "val",
                                                   transform=val_transforms,
                                                   selected_domains=self.selected_domains,
                                                   fetcher_params=self.fetcher_params)
                self.test_set = SimpleShapesDataset(self.simple_shapes_folder, "test",
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
                        [train_set, self.val_set, self.test_set])
                    self.ood_boundaries = ood_boundaries

                    target_indices = np.unique(id_ood_splits[0][0])

                    print("Val set in dist size", len(id_ood_splits[1][0]))
                    print("Val set OOD size", len(id_ood_splits[1][1]))
                    print("Test set in dist size", len(id_ood_splits[2][0]))
                    print("Test set OOD size", len(id_ood_splits[2][1]))
                else:
                    id_ood_splits = None
                    target_indices = train_set.ids

                self.val_set = split_ood_sets(self.val_set, id_ood_splits)
                self.test_set = split_ood_sets(self.test_set, id_ood_splits)

                if self.add_unimodal:
                    mapping, domain_mapping = self.filter_sync_domains(target_indices)

                    print(f"Loaded {len(target_indices)} examples in train set.")
                    self.train_set = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                                         mapping=mapping,
                                                         domain_mapping=domain_mapping,
                                                         selected_domains=self.selected_domains,
                                                         transform=train_set.transforms,
                                                         output_transform=train_set.output_transform,
                                                         fetcher_params=self.fetcher_params)
                else:
                    self.train_set = train_set

            self.set_validation_examples(
                self.train_set,
                self.val_set,
                self.test_set
            )

            # Use pre saved latents if provided.
            for shapes_set in [{"train": self.train_set}, self.val_set, self.test_set]:
                for dataset in shapes_set.values():
                    if dataset is not None:
                        if isinstance(dataset, Subset):
                            dataset = dataset.dataset
                        dataset.use_pre_saved_latents(self.pre_saved_latent_paths)

        self.is_setup = True

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
