from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch

from bim_gw.datasets.domain_loaders import PreSavedLatentLoader
from bim_gw.datasets.pre_saved_latents import load_pre_saved_latent
from bim_gw.datasets.simple_shapes.domain_loaders import (
    AttributesLoader,
    TextLoader,
)
from bim_gw.datasets.simple_shapes.types import ShapesAvailableDomains

OOODataType = Dict[
    str,
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]


class OddImageDataset:
    def __init__(
        self,
        root_path,
        split,
        pre_saved_latent_path,
        selected_domains,
        bert_latent,
    ):
        self.root_path = Path(root_path)
        self.split = split
        self.selected_domains = selected_domains

        self.labels = np.load(str(root_path / f"{split}_odd_image_labels.npy"))

        self.shift_ref_item = 0
        if split == "val":
            self.shift_ref_item = 500_000
        elif split == "test":
            self.shift_ref_item = 750_000

        ids = np.arange(1_000_000)
        labels = np.load(str(self.root_path / "train_labels.npy"))

        domain_loaders = {
            "v": lambda: PreSavedLatentLoader(
                # split always train, we used the end 500_000 as val/test
                # for this dataset.
                # We only used the 500 000 first examples for training the
                # other models, we use the 500 000 unseen elements
                # from the train set
                load_pre_saved_latent(
                    self.root_path,
                    "train",
                    pre_saved_latent_path,
                    ShapesAvailableDomains.v,
                ),
                {0: "z_img"},
            ),
            "attr": lambda: AttributesLoader(
                self.root_path, "train", ids, labels, None, use_unpaired=False
            ),
            "t": lambda: TextLoader(
                self.root_path,
                "train",
                ids,
                labels,
                None,
                bert_latents=bert_latent,
                pca_dim=768,
            ),
        }
        self.domain_loaders = {
            name: domain_loaders[name]() for name in selected_domains
        }

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, item: int) -> OOODataType:
        label = self.labels[item]
        data: OOODataType = {
            name: (
                self.domain_loaders[name].get_item(
                    label[0] + self.shift_ref_item
                ),
                self.domain_loaders[name].get_item(
                    label[1] + self.shift_ref_item
                ),
                self.domain_loaders[name].get_item(
                    label[2] + self.shift_ref_item
                ),
            )
            for name in self.domain_loaders.keys()
        }
        data["label"] = torch.tensor(label[3], dtype=torch.long)
        return data
