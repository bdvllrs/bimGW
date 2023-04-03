from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset

from bim_gw.datasets.domain import collate_fn
from bim_gw.datasets.simple_shapes.datasets import (
    AVAILABLE_DOMAINS, AvailableDomainsType,
    SimpleShapesDataset
)
from bim_gw.datasets.simple_shapes.utils import (
    create_ood_split,
    get_v_preprocess, split_ood_sets
)
from bim_gw.datasets.utils import filter_sync_domains, get_validation_examples
from bim_gw.modules.domain_modules import VAE
from bim_gw.modules.domain_modules.simple_shapes import (
    SimpleShapesAttributes,
    SimpleShapesText
)
from bim_gw.utils import registries
from bim_gw.utils.utils import get_checkpoint_path


@registries.register_domain("v")
def load_v_domain(args, im_size=None):
    return VAE.load_from_checkpoint(
        get_checkpoint_path(args.global_workspace.vae_checkpoint),
        strict=False
    )


@registries.register_domain("attr")
def load_attr_domain(args, img_size):
    return SimpleShapesAttributes(img_size)


@registries.register_domain("t")
def load_t_domain(args, img_size=None):
    return SimpleShapesText.load_from_checkpoint(
        get_checkpoint_path(args.global_workspace.lm_checkpoint),
        bert_path=args.global_workspace.bert_path,
        z_size=args.lm.z_size,
        hidden_size=args.lm.hidden_size,
        beta=args.lm.beta,
    )


class SimpleShapesDataModule(LightningDataModule):
    def __init__(
        self,
        simple_shapes_folder: str,
        batch_size: int,
        num_workers: int = 0, prop_labelled_images: float = 1.,
        prop_available_images: float = 1.,
        removed_sync_domains: Optional[List[List[str]]] = None,
        n_validation_domain_examples: int = 32,
        split_ood: bool = True,
        selected_domains: Optional[List[AvailableDomainsType]] = None,
        pre_saved_latent_paths: Optional[Dict[str, str]] = None,
        sync_uses_whole_dataset: bool = False,
        add_unimodal: bool = True,
        domain_loader_params: Optional[Dict[str, Any]] = None,
        len_train_dataset: int = 1_000_000,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ood = split_ood
        self.pre_saved_latent_paths = pre_saved_latent_paths
        self.add_unimodal = add_unimodal
        self.domain_loader_params = domain_loader_params

        self.simple_shapes_folder = Path(simple_shapes_folder)
        self.img_size: int = 32
        self.sync_uses_whole_dataset = sync_uses_whole_dataset
        self.num_channels: int = 3
        self.len_train_dataset: int = len_train_dataset
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

        ds = SimpleShapesDataset(
            simple_shapes_folder, "val",
            selected_domains=self.selected_domains,
            domain_loader_params=self.domain_loader_params
        )
        self.classes = ds.classes
        self.val_dataset_size = len(ds)

        # Remove sync for some combination of domains
        self.remove_sync_domains = removed_sync_domains

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.inception_stats_path_train = None
        self.inception_stats_path_val = None
        self.inception_stats_path_test = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ["fit", "validate", "test"]:
            val_transforms: Dict[
                AvailableDomainsType, Callable[[Any], Any]] = {
                "v": get_v_preprocess()
            }
            train_transforms: Dict[
                AvailableDomainsType, Callable[[Any], Any]] = {
                "v": get_v_preprocess()
            }
            if self.sync_uses_whole_dataset:
                sync_indices = np.arange(self.len_train_dataset)
            else:
                sync_indices = np.arange(self.len_train_dataset // 2)
            if stage == "fit" or stage is None:
                self.val_set = SimpleShapesDataset(
                    self.simple_shapes_folder, "val",
                    transform=val_transforms,
                    selected_domains=self.selected_domains,
                    domain_loader_params=self.domain_loader_params
                )
                self.test_set = SimpleShapesDataset(
                    self.simple_shapes_folder, "test",
                    transform=val_transforms,
                    selected_domains=self.selected_domains,
                    domain_loader_params=self.domain_loader_params
                )

                train_set = SimpleShapesDataset(
                    self.simple_shapes_folder, "train",
                    selected_indices=sync_indices,
                    transform=train_transforms,
                    selected_domains=self.selected_domains,
                    domain_loader_params=self.domain_loader_params
                )

                if self.split_ood:
                    id_ood_splits, ood_boundaries = create_ood_split(
                        [train_set, self.val_set, self.test_set]
                    )
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
                    mapping, domain_mapping = filter_sync_domains(
                        self.selected_domains,
                        target_indices,
                        self.prop_labelled_images,
                        self.prop_available_images,
                    )

                    self.train_set = SimpleShapesDataset(
                        self.simple_shapes_folder, "train",
                        mapping=mapping,
                        domain_mapping=domain_mapping,
                        selected_domains=self.selected_domains,
                        transform=train_set.transforms,
                        output_transform=train_set.output_transform,
                        domain_loader_params=self.domain_loader_params
                    )
                else:
                    self.train_set = train_set

            self.domain_examples = get_validation_examples(
                self.train_set,
                self.val_set,
                self.test_set,
                self.n_domain_examples,
            )

            # Use pre saved latents if provided.
            for shapes_set in [{"train": self.train_set}, self.val_set,
                               self.test_set]:
                for dataset in shapes_set.values():
                    if dataset is not None:
                        if isinstance(dataset, Subset):
                            dataset = dataset.dataset
                        dataset.use_pre_saved_latents(
                            self.pre_saved_latent_paths
                        )

    def train_dataloader(
        self, shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_set,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    def get_val_test_dataloader(
        self, dataset
    ) -> List[torch.utils.data.DataLoader]:
        dataloaders = [
            torch.utils.data.DataLoader(
                dataset["in_dist"], self.batch_size,
                num_workers=self.num_workers, pin_memory=True,
                collate_fn=collate_fn,
            ),
        ]
        if dataset["ood"] is not None:
            dataloaders.append(
                torch.utils.data.DataLoader(
                    dataset["ood"], self.batch_size,
                    num_workers=self.num_workers, pin_memory=True,
                    collate_fn=collate_fn,
                )
            )
        return dataloaders

    def val_dataloader(self) -> List[torch.utils.data.DataLoader]:
        return self.get_val_test_dataloader(self.val_set)

    def test_dataloader(self) -> List[torch.utils.data.DataLoader]:
        return self.get_val_test_dataloader(self.test_set)

    def transfer_batch_to_device(self, batch, device, dataloader_idx=None):
        for domain_items in batch.values():
            domain_items.to_device(device)
        return batch
