from pathlib import Path

import torch
from mmsdk import mmdatasdk
from pytorch_lightning import LightningDataModule

from bim_gw.datasets.cmu_mosei.dataset import CMUMOSEIDataset


class CMUMOSEIDataModule(LightningDataModule):
    def __init__(self, root_path, batch_size, num_workers, selected_domains, validate_cmu=True, seq_length=50):
        super(CMUMOSEIDataModule, self).__init__()

        self.root_path = Path(root_path)
        self.selected_domains = selected_domains
        self.cmu_dataset = mmdatasdk.mmdataset(root_path, validate=validate_cmu)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_length = seq_length

    def setup(self, stage=None):
        folds = [mmdatasdk.cmu_mosei.standard_folds.standard_train_fold,
                 mmdatasdk.cmu_mosei.standard_folds.standard_valid_fold,
                 mmdatasdk.cmu_mosei.standard_folds.standard_test_fold]

        data_folds = self.cmu_dataset.get_tensors(
            seq_len=self.seq_length,
            non_sequences=["All Labels"],
            direction=False,
            folds=folds
        )
        self.train_set = CMUMOSEIDataset(data_folds[0], "train", self.selected_domains, None)
        self.val_set = CMUMOSEIDataset(data_folds[1], "val", self.selected_domains, None)
        self.test_set = CMUMOSEIDataset(data_folds[2], "test", self.selected_domains, None)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
