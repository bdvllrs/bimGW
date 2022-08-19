from pathlib import Path

import numpy as np
import torch
from bim_gw.datasets.simple_shapes.fetchers import PreSavedLatentDataFetcher

from bim_gw.datasets.pre_saved_latents import load_pre_saved_latent


class OddImageDataset:
    def __init__(self, root_path, split, pre_saved_latent_path):
        self.root_path = Path(root_path)
        self.split = split

        self.labels = np.load(str(root_path / f"{split}_odd_image_labels.npy"))

        self.visual_domain = PreSavedLatentDataFetcher(
            # split always train, we used the end 500_000 as val/test for this dataset.
            load_pre_saved_latent(self.root_path, "train", pre_saved_latent_path, "v"))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, item):
        label = self.labels[item]
        return (
            self.visual_domain.get_item(label[0])[1],
            self.visual_domain.get_item(label[1])[1],
            self.visual_domain.get_item(label[2])[1],
            torch.tensor(label[3], dtype=torch.long)
        )
