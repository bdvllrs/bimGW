import csv
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torchvision import transforms

from bim_gw.utils.losses.compute_fid import compute_dataset_statistics


def get_preprocess(augmentation=False):
    transformations = []
    if augmentation:
        transformations.append(transforms.RandomHorizontalFlip())

    transformations.extend([
        transforms.ToTensor(),
        # transforms.Normalize(norm_mean, norm_std)
    ])

    return transforms.Compose(transformations)


class SimpleShapesDataset:
    def __init__(self, path, split="train", transforms=None):
        assert split in ["train", "val", "test"]
        self.root_path = Path(path)
        self.transforms = transforms
        self.split = split

        self.labels = []

        self.categories = OrderedDict()

        with open(self.root_path / f"labels.csv", "r") as f:
            reader = csv.reader(f)
            for k, line in enumerate(reader):
                self.categories[line[0]] = line[1:]

        with open(self.root_path / f"{split}_labels.csv", "r") as f:
            reader = csv.reader(f)
            for k, line in enumerate(reader):
                if k > 0:
                    self.labels.append(list(map(int, line)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        with open(self.root_path / self.split / f"{item}.png", 'rb') as f:
            img = Image.open(f)
        img.convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.labels[item]
        return img, label


class SimpleShapesData(LightningDataModule):
    def __init__(
            self, simple_shapes_folder, batch_size,
            num_workers=0, use_data_augmentation=False,
    ):
        super().__init__()
        self.simple_shapes_folder = Path(simple_shapes_folder)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = 32

        self.num_channels = 3
        self.use_data_augmentation = use_data_augmentation

        # self.classes = ds.classes
        ds = SimpleShapesDataset(simple_shapes_folder, "val")
        self.val_dataset_size = len(ds)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.shapes_train = SimpleShapesDataset(self.simple_shapes_folder, "train", get_preprocess(self.use_data_augmentation))
            self.shapes_val = SimpleShapesDataset(self.simple_shapes_folder, "val", get_preprocess())
        validation_reconstruction_indices = torch.randint(len(self.shapes_val), size=(self.batch_size,))
        self.validation_reconstructed_images = torch.stack([self.shapes_val[k][0]
                                                            for k in validation_reconstruction_indices], dim=0)
        if stage == "test" or stage is None:
            raise NotImplementedError

    def compute_inception_statistics(self, batch_size, device):
        train_ds = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                                get_preprocess(self.use_data_augmentation))
        val_ds = SimpleShapesDataset(self.simple_shapes_folder, "val", get_preprocess())
        self.inception_stats_path_train = compute_dataset_statistics(train_ds, self.simple_shapes_folder, "shapes_train",
                                                                     batch_size, device)
        self.inception_stats_path_val = compute_dataset_statistics(val_ds, self.simple_shapes_folder, "shapes_val",
                                                                   batch_size, device)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.shapes_train,
                                           batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.shapes_val, self.batch_size,
                                           num_workers=self.num_workers, pin_memory=True)
