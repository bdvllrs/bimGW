from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
import torch.utils.data
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import default_collate

from bim_gw.datasets.domain import domain_collate_fn
from bim_gw.datasets.odd_image.dataset import OddImageDataset
from bim_gw.modules.domain_modules.simple_shapes.attributes import (
    SimpleShapesAttributes,
)
from bim_gw.modules.domain_modules.simple_shapes.text import SimpleShapesText
from bim_gw.modules.domain_modules.vae import VAE
from bim_gw.utils import registries
from bim_gw.utils.utils import get_checkpoint_path


def collate_fn(batch: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    items: Dict[str, Any] = dict()
    for item in batch:
        for domain_name, domain_item in item.items():
            if isinstance(domain_item, tuple):
                if domain_name not in items:
                    items[domain_name] = tuple(
                        [] for _ in range(len(domain_item))
                    )
                for k in range(len(domain_item)):
                    items[domain_name][k].append(domain_item[k])
            else:
                if domain_name not in items:
                    items[domain_name] = []
                items[domain_name].append(domain_item)
    out_batch: Dict[str, Any] = {}
    for domain_name in items.keys():
        if isinstance(items[domain_name], tuple):
            out_batch[domain_name] = tuple(
                [
                    domain_collate_fn(items[domain_name][k])
                    for k in range(len(items[domain_name]))
                ]
            )
        else:
            out_batch[domain_name] = default_collate(items[domain_name])
    return out_batch


@registries.register_domain("v")
def load_v_domain(args, im_size=None):
    return VAE.load_from_checkpoint(
        get_checkpoint_path(args.global_workspace.vae_checkpoint), strict=False
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


class OddImageDataModule(LightningDataModule):
    def __init__(
        self,
        root_path,
        pre_saved_latent_path,
        batch_size,
        num_workers,
        selected_domains,
        bert_latent,
    ):
        super(OddImageDataModule, self).__init__()

        self.root_path = Path(root_path)
        self.pre_saved_latent_path = pre_saved_latent_path
        self.selected_domains = selected_domains
        self.bert_latent = bert_latent

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classes = [0, 1, 2]
        self.img_size = 32
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage=None):
        self.train_set = OddImageDataset(
            self.root_path,
            "train",
            self.pre_saved_latent_path,
            self.selected_domains,
            self.bert_latent,
        )
        self.val_set = OddImageDataset(
            self.root_path,
            "val",
            self.pre_saved_latent_path,
            self.selected_domains,
            self.bert_latent,
        )
        self.test_set = OddImageDataset(
            self.root_path,
            "test",
            self.pre_saved_latent_path,
            self.selected_domains,
            self.bert_latent,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx=None):
        for domain_key, domain_items in batch.items():
            if isinstance(domain_items, torch.Tensor):
                batch[domain_key] = domain_items.to(device)
            else:
                for domain_item in domain_items:
                    domain_item.to_device(device)
        return batch
