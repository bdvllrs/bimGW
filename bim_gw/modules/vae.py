import torch
import torchvision
from neptune.new.types import File
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F

from bim_gw.utils.losses import grad_norm


def log_image(logger, sample_imgs, name, step=None, **kwargs):
    if logger is not None:
        # sample_imgs = denormalize(sample_imgs, video_mean, video_std, clamp=True)
        sample_imgs = sample_imgs - sample_imgs.min()
        sample_imgs = sample_imgs / sample_imgs.max()
        img_grid = torchvision.utils.make_grid(sample_imgs, **kwargs)
        img_grid = torchvision.transforms.ToPILImage(mode='RGB')(img_grid.cpu())
        logger.log_image(name, File.as_image(img_grid), step)


class DeconvResNetBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, padding, upsample=False):
        super(DeconvResNetBlock, self).__init__()

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(inplanes, inplanes, kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(inplanes, planes, kernel_size, stride, 1, padding, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Identity()
        if upsample:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(inplanes, planes, kernel_size, stride, 1, padding, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.block1(x)  # (B, C, W, H)
        out = self.block2(out)  # (B, C, W, H)
        out = out + self.upsample(x)  # (B, C/2, W*2, H*2)
        out = self.relu(out)
        return out


class ResNetDecoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim

        self.layer1 = DeconvResNetBlock(512, 256, 3, 2, 1, True)
        self.layer2 = DeconvResNetBlock(256, 128, 3, 2, 1, True)
        self.layer3 = DeconvResNetBlock(128, 64, 3, 2, 1, True)
        self.layer4 = DeconvResNetBlock(64, 64, 3, 1, 0, False)

        self.unpool = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.ConvTranspose2d(64, 3, 7, 2, 3, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.linear = nn.Linear(self.z_dim, 512, bias=False)

    def forward(self, x):
        x = self.linear(x)  # (B, 512)
        x = x.reshape(x.size(0), 512, 1, 1)  # (B, 512, 1, 1)
        out = self.layer1(x)  #
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.unpool(out)
        out = self.conv(out)
        out = self.bn(out)
        return out


def reparameterize(mean, logvar):
    std = logvar.mul(0.5).exp()
    eps = torch.randn(std.size()).to(mean.device)
    return eps.mul(std).add(mean)


class VAE(LightningModule):
    """
    Adapted from https://github.com/SashaMalysheva/Pytorch-VAE
    """

    def __init__(self, image_size, channel_num, z_size, beta=1,
                 n_validation_examples=32,
                 optim_lr=3e-4, optim_weight_decay=1e-5,
                 scheduler_step=20, scheduler_gamma=0.5,
                 validation_reconstruction_images=None):
        # configurations
        super().__init__()
        self.save_hyperparameters()

        self.image_size = image_size
        self.channel_num = channel_num
        self.z_size = z_size
        self.beta = beta

        # val sampling
        self.register_buffer("validation_sampling_z", torch.randn(n_validation_examples, self.z_size))
        self.register_buffer("validation_reconstruction_images", validation_reconstruction_images)

        self.encoder = torchvision.models.resnet18(False)
        self.encoder.fc = nn.Identity()

        # q
        self.q_mean = nn.Linear(512, self.z_size)
        self.q_logvar = nn.Linear(512, self.z_size)

        # decoder
        self.decoder = ResNetDecoder(z_size)

    def forward(self, x):
        out = self.encoder(x)
        mean = self.q_mean(out)
        logvar = self.q_logvar(out)

        z = reparameterize(mean, logvar)

        # reconstruct x from z
        x_reconstructed = self.decoder(z)

        return (mean, logvar), x_reconstructed

    def reconstruction_loss(self, x_reconstructed, x):
        return F.mse_loss(x_reconstructed, x)
        # return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    def kl_divergence_loss(self, mean, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl = kl / (mean.size(0) * self.channel_num * self.image_size * self.image_size)
        return kl

    def sample(self, size):
        z = torch.randn(size, self.z_size).to(self.device)
        return self.decoder(z)

    # Lightning
    def training_step(self, batch, batch_idx):
        x, _ = batch
        (mean, logvar), x_reconstructed = self(x)
        reconstruction_loss = self.reconstruction_loss(x_reconstructed, x)
        kl_divergence_loss = self.kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + self.beta * kl_divergence_loss

        # self.log("train_mse_grads", grad_norm(reconstruction_loss, self.parameters(), retain_graph=True, allow_unused=True))
        # self.log("train_kl_grads", grad_norm(kl_divergence_loss, self.parameters(), retain_graph=True, allow_unused=True))

        self.log("train_reconstruction_loss", reconstruction_loss, logger=True)
        self.log("train_kl_divergence_loss", kl_divergence_loss, logger=True)
        self.log("train_total_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        (mean, logvar), x_reconstructed = self(x)
        reconstruction_loss = self.reconstruction_loss(x_reconstructed, x)
        kl_divergence_loss = self.kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + kl_divergence_loss

            # self.log("val_mse_grads", grad_norm(reconstruction_loss, self.parameters(), retain_graph=True, allow_unused=True))
            # self.log("val_kl_grads", grad_norm(kl_divergence_loss, self.parameters(), allow_unused=True))

        self.log("val_reconstruction_loss", reconstruction_loss, on_epoch=True)
        self.log("val_kl_divergence_loss", kl_divergence_loss, on_epoch=True)
        self.log("val_total_loss", total_loss, on_epoch=True)

    def validation_epoch_end(self, outputs):
        x = self.validation_reconstruction_images
        _, x_reconstructed = self(x)

        if self.current_epoch == 0:
            log_image(self.logger, x[:self.hparams.n_validation_examples], "val_original_images")

        log_image(self.logger, x_reconstructed[:self.hparams.n_validation_examples], "val_reconstruction")
        sampled_images = self.decoder(self.validation_sampling_z)
        log_image(self.logger, sampled_images, "val_sampling")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optim_lr,
                                     weight_decay=self.hparams.optim_weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.scheduler_step, self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]
