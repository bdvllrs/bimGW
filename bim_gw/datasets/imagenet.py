import torch
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torchvision.datasets import ImageNet

norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_preprocess(img_size):
    # TODO: data augmentation
    return transforms.Compose([
        transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])


class ImageNetData(LightningDataModule):
    def __init__(self, imagenet_folder, batch_size, img_size=32, num_workers=0):
        super().__init__()
        self.image_net_folder = imagenet_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.num_channels = 3

        self.image_net_train = ImageNet(self.image_net_folder, transform=get_preprocess(self.img_size))
        self.image_net_val = ImageNet(self.image_net_folder, transform=get_preprocess(self.img_size), split="val")

        validation_reconstruction_indices = torch.randint(len(self.image_net_val), size=(batch_size,))
        self.validation_reconstructed_images = torch.stack([self.image_net_val[k][0] for k in validation_reconstruction_indices], dim=0)

    def train_dataloader(self):
        dataloader_train = torch.utils.data.DataLoader(self.image_net_train,
                                                       batch_size=self.batch_size, shuffle=True,
                                                       num_workers=self.num_workers, pin_memory=True)
        return dataloader_train

    def val_dataloader(self):
        dataloader_val = torch.utils.data.DataLoader(self.image_net_val, self.batch_size,
                                                     num_workers=self.num_workers, pin_memory=True)
        return dataloader_val
