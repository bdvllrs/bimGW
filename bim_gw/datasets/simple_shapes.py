import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset
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
    def __init__(self, path, split="train", transform=None, output_transform=None,
                 selected_indices=None, min_dataset_size=None):
        """
        Args:
            path:
            split:
            transform:
            output_transform:
            selected_indices: To reduce the size of the dataset to given indices.
            min_dataset_size: Copies the data so that the effective size is the one given.
        """
        assert split in ["train", "val", "test"]
        self.root_path = Path(path)
        self.transforms = transform
        self.output_transform = output_transform
        self.split = split
        self.img_size = 32

        self.classes = np.array(["square", "circle", "triangle"])
        self.labels = []
        self.ids = []

        with open(self.root_path / f"{split}_labels.csv", "r") as f:
            reader = csv.reader(f)
            for k, line in enumerate(reader):
                if k > 0 and (selected_indices is None or k in selected_indices):
                    self.labels.append(list(map(float, line)))
                    self.ids.append(k - 1)

        self.ids = np.array(self.ids)
        self.labels = np.array(self.labels, dtype=np.float32)

        if min_dataset_size is not None:
            original_size = len(self.labels)
            n_repeats = min_dataset_size // original_size + 1 * int(min_dataset_size % original_size > 0)
            self.ids = np.tile(self.ids, n_repeats)
            self.labels = np.tile(self.labels, (n_repeats, 1))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image_id = self.ids[item]
        with open(self.root_path / self.split / f"{image_id}.png", 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.labels[item]
        cls = int(label[0])
        x, y = label[1], label[2]
        radius = label[3]
        # if cls == 0:  # square
        #     rotation = label[4] % 90
        # elif cls == 1:  # circle
        #     rotation = 0
        # else:
        rotation = label[4]
        # assert 0 <= rotation <= 1
        # rotation = rotation * 2 * np.pi / 360  # put in radians
        r, g, b = label[5], label[6], label[7]
        rotation_x = (np.cos(rotation) + 1) / 2
        rotation_y = (np.sin(rotation) + 1) / 2

        labels = [
            cls,
            torch.tensor([x, y, radius, rotation_x, rotation_y, r, g, b], dtype=torch.float),
        ]

        if self.output_transform is not None:
            return self.output_transform(img, labels)
        return img, labels


class SimpleShapesData(LightningDataModule):
    def __init__(
            self, simple_shapes_folder, batch_size,
            num_workers=0, use_data_augmentation=False, prop_labelled_images=1.,
            n_validation_domain_examples=None,
            bimodal=False,
    ):
        super().__init__()
        if bimodal and use_data_augmentation:
            raise ValueError("bimodal mode and data augmentation is not possible for now...")
        self.simple_shapes_folder = Path(simple_shapes_folder)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = 32
        self.bimodal = bimodal
        self.validation_domain_examples = n_validation_domain_examples if n_validation_domain_examples is not None else batch_size

        assert 0 <= prop_labelled_images <= 1, "The proportion of labelled images must be between 0 and 1."
        self.prop_labelled_images = prop_labelled_images

        self.num_channels = 3
        self.use_data_augmentation = use_data_augmentation

        ds = SimpleShapesDataset(simple_shapes_folder, "val")
        self.classes = ds.classes
        self.val_dataset_size = len(ds)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.bimodal:
                sync_set = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                               get_preprocess(self.use_data_augmentation),
                                               lambda v, t: {"v": v, "t": t})

                v_train_set = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                                  get_preprocess(self.use_data_augmentation),
                                                  lambda v, t: v)

                if self.prop_labelled_images < 1.:
                    # Unlabel randomly some elements
                    n_targets = len(v_train_set)
                    target_indices = np.arange(n_targets)
                    np.random.shuffle(target_indices)
                    num_unlabelled = int((1 - self.prop_labelled_images) * n_targets)
                    labelled_elems = target_indices[num_unlabelled:]
                    print(f"Training using {len(labelled_elems)} labelled examples.")
                    sync_set = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                                   get_preprocess(self.use_data_augmentation),
                                                   lambda v, t: {"v": v, "t": t},
                                                   labelled_elems, n_targets)

                self.train_datasets = {
                    "v": v_train_set,
                    "t": SimpleShapesDataset(self.simple_shapes_folder, "train",
                                             get_preprocess(self.use_data_augmentation),
                                             lambda v, t: t),
                    "sync_": sync_set
                }
            else:
                self.shapes_train = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                                        get_preprocess(self.use_data_augmentation))

        if self.bimodal:
            self.shapes_val = SimpleShapesDataset(self.simple_shapes_folder, "val", get_preprocess(),
                                                  lambda v, t: {"v": v, "t": t})
            self.shapes_test = SimpleShapesDataset(self.simple_shapes_folder, "test", get_preprocess(),
                                                   lambda v, t: {"v": v, "t": t})
            visual_index = "v"
            text_index = "t"
        else:
            self.shapes_val = SimpleShapesDataset(self.simple_shapes_folder, "val", get_preprocess())
            self.shapes_test = SimpleShapesDataset(self.simple_shapes_folder, "test", get_preprocess())
            visual_index = 0
            text_index = 1

        test_set = self.shapes_val if stage == "fit" else self.shapes_test

        validation_reconstruction_indices = torch.randint(len(test_set), size=(self.validation_domain_examples,))

        self.validation_domain_examples = {
            "v": torch.stack([test_set[k][visual_index] for k in validation_reconstruction_indices], dim=0),
            "t": []
        }

        # add t examples
        for k in range(len(test_set[0][text_index])):
            if isinstance(test_set[0][text_index][k], (int, float)):
                self.validation_domain_examples["t"].append(
                    torch.tensor([test_set[i][text_index][k] for i in validation_reconstruction_indices])
                )
            else:
                self.validation_domain_examples["t"].append(
                    torch.stack([test_set[i][text_index][k] for i in validation_reconstruction_indices], dim=0)
                )

    def compute_inception_statistics(self, batch_size, device):
        train_ds = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                       get_preprocess(self.use_data_augmentation))
        val_ds = SimpleShapesDataset(self.simple_shapes_folder, "val", get_preprocess())
        test_ds = SimpleShapesDataset(self.simple_shapes_folder, "test", get_preprocess())
        self.inception_stats_path_train = compute_dataset_statistics(train_ds, self.simple_shapes_folder,
                                                                     "shapes_train",
                                                                     batch_size, device)
        self.inception_stats_path_val = compute_dataset_statistics(val_ds, self.simple_shapes_folder, "shapes_val",
                                                                   batch_size, device)

        self.inception_stats_path_test = compute_dataset_statistics(test_ds, self.simple_shapes_folder, "shapes_test",
                                                                    batch_size, device)

    def train_dataloader(self):
        if self.bimodal:
            dataloaders = {}
            for key, dataset in self.train_datasets.items():
                # dataloaders[key] = torch.utils.data.DataLoader(Subset(dataset, torch.arange(0, 2 * self.batch_size)),
                dataloaders[key] = torch.utils.data.DataLoader(dataset,
                                                               batch_size=self.batch_size, shuffle=True,
                                                               num_workers=self.num_workers, pin_memory=True)
            return dataloaders
        return torch.utils.data.DataLoader(self.shapes_train,
                                           batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.shapes_val, self.batch_size,
                                           num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.shapes_test, self.batch_size,
                                           num_workers=self.num_workers, pin_memory=True)
