from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule

from bim_gw.datasets.odd_image.dataset import OddImageDataset


class OddImageDataModule(LightningDataModule):
    def __init__(self, root_path, pre_saved_latent_path, batch_size, num_workers):
        super(OddImageDataModule, self).__init__()

        self.root_path = Path(root_path)
        self.pre_saved_latent_path = pre_saved_latent_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classes = [0, 1, 2]
        self.img_size = 32

    def setup(self, stage=None):
        self.train_set = OddImageDataset(self.root_path, "train", self.pre_saved_latent_path)
        self.val_set = OddImageDataset(self.root_path, "val", self.pre_saved_latent_path)
        self.test_set = OddImageDataset(self.root_path, "test", self.pre_saved_latent_path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           pin_memory=True)
