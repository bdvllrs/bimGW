import torch
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torchvision.datasets import CIFAR10

from bim_gw.utils.losses.compute_fid import compute_dataset_statistics

norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def get_preprocess(augmentation=False):
    transformations = []
    if augmentation:
        transformations.append(transforms.RandomHorizontalFlip())

    transformations.extend([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    return transforms.Compose(transformations)


class CIFARData(LightningDataModule):
    def __init__(
            self, cifar_folder, batch_size,
            num_workers=0, use_data_augmentation=False,
    ):
        super().__init__()
        self.cifar_folder = cifar_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = 32

        self.num_channels = 3
        self.use_data_augmentation = use_data_augmentation

        ds = CIFAR10(self.cifar_folder, train=False, download=True)
        self.classes = ds.classes
        self.val_dataset_size = len(ds)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.cifar_train = CIFAR10(self.cifar_folder, train=True, download=True,
                                       transform=get_preprocess(self.use_data_augmentation))
            self.cifar_val = CIFAR10(self.cifar_folder, train=False, download=True,
                                     transform=get_preprocess())
        validation_reconstruction_indices = torch.randint(len(self.cifar_val), size=(self.batch_size,))
        self.validation_reconstructed_images = torch.stack([self.cifar_val[k][0]
                                                            for k in validation_reconstruction_indices], dim=0)
        if stage == "test" or stage is None:
            raise NotImplementedError

    def compute_inception_statistics(self, batch_size, device):
        train_ds = CIFAR10(self.cifar_folder, transform=get_preprocess(), train=True)
        val_ds = CIFAR10(self.cifar_folder, transform=get_preprocess(), train=False)
        self.inception_stats_path_train = compute_dataset_statistics(train_ds, self.cifar_folder, "cifar_train",
                                                                     batch_size, device)
        self.inception_stats_path_val = compute_dataset_statistics(val_ds, self.cifar_folder, "cifar_val",
                                                                   batch_size, device)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_train,
                                           batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_val, self.batch_size,
                                           num_workers=self.num_workers, pin_memory=True)
