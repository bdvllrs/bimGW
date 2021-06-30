import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torchvision.datasets import ImageNet

norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_preprocess(img_size, augmentation=False):
    transformations = [
        transforms.Resize(img_size * 2 if augmentation else img_size),
    ]

    if augmentation:
        transformations.append(transforms.RandomResizedCrop(img_size))
        transformations.append(transforms.RandomHorizontalFlip())
    else:
        transformations.append(transforms.CenterCrop(img_size))

    transformations.extend([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    return transforms.Compose(transformations)


class ParallelDatasets(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super(ParallelDatasets, self).__init__()
        self.datasets = datasets
        self.lengths = list(map(len, datasets))
        self.len = max(self.lens)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        samples = []
        for l, dataset in zip(self.lenths, self.datasets):
            samples.append(dataset[item % l])
        return samples


class BimodalImageNet(ImageNet):
    def __init__(self, path, split="train", transform=None):
        super(BimodalImageNet, self).__init__(path, split, transform=transform)

    def __getitem__(self, item):
        img, target = super(BimodalImageNet, self).__getitem__(item)
        return img, self.classes[target]

    @staticmethod
    def collate(batch):
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, list(map(lambda x: x[0], targets))


class UnimodalImageNet(ImageNet):
    def __init__(self, path, modality, split="train", transform=None):
        super(UnimodalImageNet, self).__init__(path, split, transform=transform)
        assert modality in ["v", "t"]
        modality_index = {"v": 0, "t": 1}
        self.modality = modality_index[modality]

    def __getitem__(self, item):
        output = super(UnimodalImageNet, self).__getitem__(item)
        return output[self.modality]


class SyncImageNet(ImageNet):
    def __init__(self, path, split="train", transform=None):
        super(SyncImageNet, self).__init__(path, split, transform=transform)

    def __getitem__(self, item):
        image, target = super(SyncImageNet, self).__getitem__(item)
        return {"v": image, "t": target}


class ImageNetData(LightningDataModule):
    def __init__(
            self, imagenet_folder, batch_size,
            img_size=32, num_workers=0,
            use_data_augmentation=False,
            prop_labelled_images=1., classes_labelled_images=None,
            bimodal=False
    ):
        super().__init__()
        self.image_net_folder = imagenet_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.bimodal = bimodal

        assert prop_labelled_images >= 0, "The proportion of labelled images cannot be negative."
        self.prop_labelled_images = prop_labelled_images
        self.classes_labelled_images = classes_labelled_images
        self.num_channels = 3

        if bimodal:
            synchronized_imagenet = SyncImageNet(self.image_net_folder,
                                                 transform=get_preprocess(self.img_size, use_data_augmentation))

            if self.classes_labelled_images is None:
                self.classes_labelled_images = np.arange(len(synchronized_imagenet.classes))

            if self.prop_labelled_images < 1:
                # Unlabel some classes
                targets = np.array(synchronized_imagenet.targets)
                labelled_elems = np.isin(targets, self.classes_labelled_images)

                # Unlabel randomly some elements
                labelled_elems = np.where(labelled_elems)[0]
                n_targets = len(labelled_elems)
                target_indices = np.arange(n_targets)
                np.random.shuffle(target_indices)
                num_unlabelled = int((1 - self.prop_labelled_images) * n_targets)
                labelled_elems = labelled_elems[target_indices[num_unlabelled:]]
                synchronized_imagenet = torch.utils.data.Subset(synchronized_imagenet, labelled_elems)

            self.train_datasets = {
                "v": UnimodalImageNet(self.image_net_folder, "v",
                                      transform=get_preprocess(self.img_size, use_data_augmentation)),
                "t": UnimodalImageNet(self.image_net_folder, "t",
                                      transform=get_preprocess(self.img_size, use_data_augmentation)),
                "sync_": synchronized_imagenet
            }
        else:
            self.image_net_train = ImageNet(self.image_net_folder,
                                            transform=get_preprocess(self.img_size, use_data_augmentation))

        if self.bimodal:
            self.image_net_val = SyncImageNet(self.image_net_folder, transform=get_preprocess(self.img_size),
                                              split="val")
            visual_index = "v"
        else:
            self.image_net_val = ImageNet(self.image_net_folder, transform=get_preprocess(self.img_size),
                                          split="val")
            visual_index = 0

        validation_reconstruction_indices = torch.randint(len(self.image_net_val), size=(batch_size,))
        self.validation_reconstructed_images = torch.stack([self.image_net_val[k][visual_index]
                                                            for k in validation_reconstruction_indices], dim=0)

    def train_dataloader(self):
        if self.bimodal:
            dataloaders = {}
            for key, dataset in self.train_datasets.items():
                dataloaders[key] = torch.utils.data.DataLoader(dataset,
                                                               batch_size=self.batch_size, shuffle=True,
                                                               num_workers=self.num_workers, pin_memory=True)
            return dataloaders
        else:
            return torch.utils.data.DataLoader(self.image_net_train,
                                               batch_size=self.batch_size, shuffle=True,
                                               num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.image_net_val, self.batch_size,
                                           num_workers=self.num_workers, pin_memory=True)
