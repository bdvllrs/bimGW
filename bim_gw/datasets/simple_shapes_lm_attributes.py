from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningDataModule

from bim_gw.datasets.simple_shapes import SimpleShapesDataset, get_preprocess


class SimpleShapesLMToAttributesDataset(SimpleShapesDataset):
    def __init__(self, path, split="train", transform=None, output_transform=None):
        super().__init__(path, split, transform, output_transform, textify=True)

    def __getitem__(self, item):
        label = self.labels[item]
        cls = int(label[0])
        x, y = label[1], label[2]
        size = label[3]
        rotation = label[4]
        # assert 0 <= rotation <= 1
        # rotation = rotation * 2 * np.pi / 360  # put in radians
        r, g, b = label[5] / 255, label[6] / 255, label[7] / 255
        rotation_x = (np.cos(rotation) + 1) / 2
        rotation_y = (np.sin(rotation) + 1) / 2

        sentence = self.text_composer({
            "shape": cls,
            "rotation": rotation,
            "color": (label[5], label[6], label[7]),
            "size": size,
            "location": (x, y)
        })

        labels = (
            cls,
            torch.tensor([x, y, size, rotation_x, rotation_y, r, g, b], dtype=torch.float),
        )

        if self.output_transform is not None:
            return self.output_transform(sentence, labels)
        return sentence, labels


class SimpleShapesData(LightningDataModule):
    def __init__(
            self, simple_shapes_folder, batch_size, num_workers=0,
            n_validation_domain_examples=None,
    ):
        super().__init__()
        self.simple_shapes_folder = Path(simple_shapes_folder)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = 32
        self.num_channels = 3
        self.n_validation_domain_examples = n_validation_domain_examples if n_validation_domain_examples is not None else batch_size

        ds = SimpleShapesLMToAttributesDataset(simple_shapes_folder, "val")
        self.classes = ds.classes
        self.val_dataset_size = len(ds)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.shapes_train = SimpleShapesLMToAttributesDataset(self.simple_shapes_folder, "train", get_preprocess())
            self.shapes_val = SimpleShapesLMToAttributesDataset(self.simple_shapes_folder, "val", get_preprocess())
        self.shapes_test = SimpleShapesLMToAttributesDataset(self.simple_shapes_folder, "test", get_preprocess())

        validation_reconstruction_indices = torch.randint(len(self.shapes_val),
                                                          size=(self.n_validation_domain_examples,))
        self.validation_domain_examples = {
            "s": [self.shapes_val[k][0] for k in validation_reconstruction_indices],
            "a": [],
        }
        for i in range(len(self.shapes_val[0][1])):
            examples = [self.shapes_val[k][1][i] for k in validation_reconstruction_indices]
            if isinstance(self.shapes_val[0][1][i], (int, float)):
                self.validation_domain_examples["a"].append(torch.tensor(examples))
            else:
                self.validation_domain_examples["a"].append(torch.stack(examples, dim=0))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.shapes_train,
                                           batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.shapes_val,
                                           batch_size=self.batch_size, shuffle=False,
                                           num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.shapes_test,
                                           batch_size=self.batch_size, shuffle=False,
                                           num_workers=self.num_workers, pin_memory=True)
