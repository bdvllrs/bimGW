import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset
from torchvision import transforms
from transformers import BertTokenizer
import matplotlib.pyplot as plt

from bim_gw.utils.losses.compute_fid import compute_dataset_statistics
from bim_gw.utils.text_composer.composer import Composer
from bim_gw.utils.text_composer.writers import writers


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
                 selected_indices=None, min_dataset_size=None, textify=False):
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
                if k > 0 and (selected_indices is None or k - 1 in selected_indices):
                    self.labels.append(list(map(float, line)))
                    self.ids.append(k - 1)

        self.ids = np.array(self.ids)
        self.labels = np.array(self.labels, dtype=np.float32)

        if min_dataset_size is not None:
            original_size = len(self.labels)
            n_repeats = min_dataset_size // original_size + 1 * int(min_dataset_size % original_size > 0)
            self.ids = np.tile(self.ids, n_repeats)
            self.labels = np.tile(self.labels, (n_repeats, 1))

        self.text_composer = None
        if textify:
            self.text_composer = Composer(writers)

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
        size = label[3]
        # if cls == 0:  # square
        #     rotation = label[4] % 90
        # elif cls == 1:  # circle
        #     rotation = 0
        # else:
        rotation = label[4]
        # assert 0 <= rotation <= 1
        # rotation = rotation * 2 * np.pi / 360  # put in radians
        r, g, b = label[5] / 255, label[6] / 255, label[7] / 255
        rotation_x = (np.cos(rotation) + 1) / 2
        rotation_y = (np.sin(rotation) + 1) / 2

        if self.text_composer is not None:
            sentence = self.text_composer({
                "shape": cls,
                "rotation": rotation,
                "color": (label[5], label[6], label[7]),
                "size": size,
                "location": (x, y)
            })
            labels = sentence

        else:
            labels = (
                cls,
                torch.tensor([x, y, size, rotation_x, rotation_y, r, g, b], dtype=torch.float),
            )

        if self.output_transform is not None:
            return self.output_transform(img, labels)
        return img, labels


class SimpleShapesData(LightningDataModule):
    def __init__(
            self, simple_shapes_folder, batch_size,
            num_workers=0, use_data_augmentation=False, prop_labelled_images=1.,
            n_validation_domain_examples=None,
            bimodal=False, split_ood=True, textify=False
    ):
        super().__init__()
        if bimodal and use_data_augmentation:
            raise ValueError("bimodal mode and data augmentation is not possible for now...")
        self.simple_shapes_folder = Path(simple_shapes_folder)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = 32
        self.bimodal = bimodal
        self.split_ood = split_ood
        self.validation_domain_examples = n_validation_domain_examples if n_validation_domain_examples is not None else batch_size
        self.ood_boundaries = None
        self.textify = textify

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
                self.shapes_val = SimpleShapesDataset(self.simple_shapes_folder, "val", get_preprocess(),
                                                      lambda v, t: {"v": v, "t": t}, textify=self.textify)
                self.shapes_test = SimpleShapesDataset(self.simple_shapes_folder, "test", get_preprocess(),
                                                       lambda v, t: {"v": v, "t": t}, textify=self.textify)
                visual_index = "v"
                text_index = "t"

                sync_set = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                               get_preprocess(self.use_data_augmentation),
                                               lambda v, t: {"v": v, "t": t}, textify=self.textify)

                v_train_set = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                                  get_preprocess(self.use_data_augmentation),
                                                  lambda v, t: v)

                if self.split_ood:
                    id_ood_splits, ood_boundaries = create_ood_split([sync_set, self.shapes_val, self.shapes_test])
                    self.ood_boundaries = ood_boundaries

                    self.shapes_val = {
                        "in_dist": torch.utils.data.Subset(self.shapes_val, id_ood_splits[1][0]),
                        "ood": torch.utils.data.Subset(self.shapes_val, id_ood_splits[1][1]),
                    }
                    self.shapes_test = {
                        "in_dist": torch.utils.data.Subset(self.shapes_test, id_ood_splits[2][0]),
                        "ood": torch.utils.data.Subset(self.shapes_test, id_ood_splits[2][1]),
                    }

                    target_indices = np.unique(id_ood_splits[0][0])

                    print("Val set in dist size", len(id_ood_splits[1][0]))
                    print("Val set OOD size", len(id_ood_splits[1][1]))
                    print("Test set in dist size", len(id_ood_splits[2][0]))
                    print("Test set OOD size", len(id_ood_splits[2][1]))
                else:
                    self.shapes_val = {
                        "in_dist": self.shapes_val,
                        "ood": None
                    }
                    self.shapes_test = {
                        "in_dist": self.shapes_test,
                        "ood": None
                    }
                    target_indices = np.arange(len(v_train_set))

                if self.prop_labelled_images < 1.:
                    # Unlabel randomly some elements
                    n_targets = len(v_train_set)
                    np.random.shuffle(target_indices)
                    num_labelled = int(self.prop_labelled_images * n_targets)
                    labelled_elems = target_indices[:num_labelled]
                    print(f"Training using {len(labelled_elems)} labelled examples.")
                    sync_set = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                                   get_preprocess(self.use_data_augmentation),
                                                   lambda v, t: {"v": v, "t": t},
                                                   labelled_elems, n_targets, textify=self.textify)

                self.train_datasets = {
                    "v": v_train_set,
                    "t": SimpleShapesDataset(self.simple_shapes_folder, "train",
                                             get_preprocess(self.use_data_augmentation),
                                             lambda v, t: t,
                                             textify=self.textify),
                    "sync_": sync_set
                }
            else:
                self.shapes_val = SimpleShapesDataset(self.simple_shapes_folder, "val", get_preprocess(),
                                                      textify=self.textify)
                self.shapes_test = SimpleShapesDataset(self.simple_shapes_folder, "test", get_preprocess(),
                                                       textify=self.textify)
                visual_index = 0
                text_index = 1
                self.shapes_train = SimpleShapesDataset(self.simple_shapes_folder, "train",
                                                        get_preprocess(self.use_data_augmentation),
                                                        textify=self.textify)

        test_set = self.shapes_val if stage == "fit" else self.shapes_test
        validation_reconstruction_indices = {}
        validation_reconstruction_indices["in_dist"] = torch.randint(len(test_set["in_dist"]),
                                                                     size=(self.validation_domain_examples,))
        validation_reconstruction_indices["ood"] = None

        if test_set["ood"] is not None:
            validation_reconstruction_indices["ood"] = torch.randint(len(test_set["ood"]),
                                                                     size=(self.validation_domain_examples,))

        self.validation_domain_examples = {
            "in_dist": {
                "v": torch.stack(
                    [test_set["in_dist"][k][visual_index] for k in validation_reconstruction_indices["in_dist"]],
                    dim=0),
                "t": [],
            },
            "ood": {
                "v": None if test_set["ood"] is None else torch.stack(
                    [test_set["ood"][k][visual_index] for k in validation_reconstruction_indices["ood"]], dim=0),
                "t": None if test_set["ood"] is None else []
            }
        }

        # add t examples
        for used_dist in ["in_dist", "ood"]:
            if validation_reconstruction_indices[used_dist] is not None:
                if self.textify:
                    examples = [test_set[used_dist][i][text_index] for i in
                                validation_reconstruction_indices[used_dist]]
                    self.validation_domain_examples[used_dist]["t"] = examples
                else:
                    for k in range(len(test_set[used_dist][0][text_index])):
                        examples = [test_set[used_dist][i][text_index][k] for i in
                                    validation_reconstruction_indices[used_dist]]
                        if isinstance(test_set[used_dist][0][text_index][k], (int, float)):
                            self.validation_domain_examples[used_dist]["t"].append(
                                torch.tensor(examples)
                            )
                        else:
                            self.validation_domain_examples[used_dist]["t"].append(
                                torch.stack(examples, dim=0)
                            )
                    self.validation_domain_examples[used_dist]["t"] = tuple(
                        self.validation_domain_examples[used_dist]["t"])

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
        dataloaders = [
            torch.utils.data.DataLoader(self.shapes_val["in_dist"], self.batch_size,
                                        num_workers=self.num_workers, pin_memory=True),
        ]
        if self.shapes_val["ood"] is not None:
            dataloaders.append(
                torch.utils.data.DataLoader(self.shapes_val["ood"], self.batch_size,
                                            num_workers=self.num_workers, pin_memory=True)
            )
        return dataloaders

    def test_dataloader(self):
        dataloaders = [
            torch.utils.data.DataLoader(self.shapes_test["in_dist"], self.batch_size,
                                        num_workers=self.num_workers, pin_memory=True),
        ]

        if self.shapes_test["ood"] is not None:
            dataloaders.append(
                torch.utils.data.DataLoader(self.shapes_test["ood"], self.batch_size,
                                            num_workers=self.num_workers, pin_memory=True)
            )
        return dataloaders


def in_interval(x, xmin, xmax, val_min, val_max):
    if val_min <= xmin <= xmax <= val_max:
        return xmin <= x <= xmax
    if xmax < xmin:
        return xmin <= x <= val_max or val_min <= x <= xmax
    return False


def split_in_out_dist(dataset, ood_attrs, shape_boundaries, color_boundaries, size_boundaries,
                      rotation_boundaries, x_boundaries, y_boundaries):
    in_dist_items = []
    out_dist_imates = []

    for i, label in enumerate(dataset.labels):
        cls = int(label[0])
        x, y = label[1], label[2]
        size = label[3]
        rotation = label[4]
        hue = label[8]
        keep = True
        for k in range(3):
            n_cond_checked = 0
            if "shape" in ood_attrs[k] and shape_boundaries[k] == cls:
                n_cond_checked += 1
            if "position" in ood_attrs[k] and in_interval(x, x_boundaries[k], x_boundaries[(k + 1) % 3], 0, 32):
                n_cond_checked += 1
            if "position" in ood_attrs[k] and in_interval(y, y_boundaries[k], y_boundaries[(k + 1) % 3], 0, 32):
                n_cond_checked += 1
            if "color" in ood_attrs[k] and in_interval(hue, color_boundaries[k], color_boundaries[(k + 1) % 3], 0, 256):
                n_cond_checked += 1
            if "size" in ood_attrs[k] and in_interval(size, size_boundaries[k], size_boundaries[(k + 1) % 3], 0, 25):
                n_cond_checked += 1
            if "rotation" in ood_attrs[k] and in_interval(rotation, rotation_boundaries[k],
                                                          rotation_boundaries[(k + 1) % 3],
                                                          0, 2 * np.pi):
                n_cond_checked += 1
            if n_cond_checked >= len(ood_attrs[k]):
                keep = False
                break
        if keep:
            in_dist_items.append(i)
        else:
            out_dist_imates.append(i)
    return in_dist_items, out_dist_imates


def create_ood_split(datasets):
    """
    Splits the space of each attribute in 3 and hold one ood part
    Args:
        datasets:
    Returns:
    """
    shape_boundary = random.randint(0, 2)
    shape_boundaries = shape_boundary, (shape_boundary + 1) % 3, (shape_boundary + 2) % 3

    color_boundary = random.randint(0, 255)
    color_boundaries = color_boundary, (color_boundary + 85) % 256, (color_boundary + 170) % 256

    size_boundary = random.randint(10, 25)
    size_boundaries = size_boundary, 10 + (size_boundary - 5) % 16, 10 + size_boundary % 16

    rotation_boundary = random.random() * 2 * np.pi
    rotation_boundaries = rotation_boundary, (rotation_boundary + 2 * np.pi / 3) % (2 * np.pi), (
            rotation_boundary + 4 * np.pi / 3) % (2 * np.pi)

    x_boundary = random.random() * 32
    x_boundaries = x_boundary, (x_boundary + 11.6) % 32, (x_boundary + 23.2) % 32

    y_boundary = random.random() * 32
    y_boundaries = y_boundary, (y_boundary + 11.6) % 32, (y_boundary + 23.2) % 32

    print("boundaries")
    print("shape", shape_boundaries)
    print("color", color_boundaries)
    print("size", size_boundaries)
    print("rotation", rotation_boundaries)
    print("x", x_boundaries)
    print("y", y_boundaries)

    selected_attributes = []
    for i in range(3):
        holes_k = []
        choices = ["position", "rotation", "size", "color", "shape"]
        for k in range(2):
            choice = random.randint(0, len(choices) - 1)
            holes_k.append(choices[choice])
            choices.pop(choice)
        selected_attributes.append(holes_k)
    print("OOD: ", selected_attributes)

    ood_boundaries = {
        "x": x_boundaries,
        "y": y_boundaries,
        "size": size_boundaries,
        "rotation": rotation_boundaries,
        "color": color_boundaries,
        "selected_attributes": selected_attributes
    }

    out_datasets = []
    for dataset in datasets:
        in_dist, out_dist = split_in_out_dist(dataset, selected_attributes, shape_boundaries, color_boundaries,
                                              size_boundaries, rotation_boundaries, x_boundaries, y_boundaries)
        out_datasets.append((in_dist, out_dist))
    return out_datasets, ood_boundaries
