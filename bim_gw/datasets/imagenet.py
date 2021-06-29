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


class ImageNetData(LightningDataModule):
    def __init__(self, imagenet_folder, batch_size, img_size=32, num_workers=0, use_data_augmentation=False,
                 bimodal=False):
        super().__init__()
        self.image_net_folder = imagenet_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.num_channels = 3

        imagenet_class = BimodalImageNet if bimodal else ImageNet
        self.collate_fn = BimodalImageNet.collate if bimodal else None

        self.image_net_train = imagenet_class(self.image_net_folder,
                                              transform=get_preprocess(self.img_size, use_data_augmentation))
        self.image_net_val = imagenet_class(self.image_net_folder, transform=get_preprocess(self.img_size),
                                            split="val")

        validation_reconstruction_indices = torch.randint(len(self.image_net_val), size=(batch_size,))
        self.validation_reconstructed_images = torch.stack([self.image_net_val[k][0]
                                                            for k in validation_reconstruction_indices], dim=0)

    def train_dataloader(self):
        dataloader_train = torch.utils.data.DataLoader(self.image_net_train,
                                                       batch_size=self.batch_size, shuffle=True,
                                                       collate_fn=self.collate_fn,
                                                       num_workers=self.num_workers, pin_memory=True)
        return dataloader_train

    def val_dataloader(self):
        dataloader_val = torch.utils.data.DataLoader(self.image_net_val, self.batch_size,
                                                     collate_fn=self.collate_fn,
                                                     num_workers=self.num_workers, pin_memory=True)
        return dataloader_val
