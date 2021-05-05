import torch
import torchvision
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule


def log_image(logger, sample_imgs, name, step=None, **kwargs):
    if logger is not None:
        # sample_imgs = denormalize(sample_imgs, video_mean, video_std, clamp=True)
        img_grid = torchvision.utils.make_grid(sample_imgs, **kwargs)
        img_grid = torchvision.transforms.ToPILImage(mode='RGB')(img_grid.cpu())
        logger.log_image(name, img_grid, step)


class VAE(LightningModule):
    """
    Adapted from https://github.com/SashaMalysheva/Pytorch-VAE
    """
    def __init__(self, image_size, channel_num, kernel_num, z_size,
                 n_validation_examples=32,
                 optim_lr=3e-4, optim_weight_decay=1e-5):
        # configurations
        super().__init__()
        self.save_hyperparameters()

        self.image_size = image_size
        self.channel_num = channel_num
        self.kernel_num = kernel_num
        self.z_size = z_size

        # val sampling
        self.register_buffer("validation_sampling_z", torch.randn(n_validation_examples, self.z_size))


        # encoder
        self.encoder = nn.Sequential(
            self._conv(self.channel_num, self.kernel_num // 4),
            self._conv(self.kernel_num // 4, self.kernel_num // 2),
            self._conv(self.kernel_num // 2, self.kernel_num),
        )

        # encoded feature's size and volume
        self.feature_size = self.image_size // 8
        self.feature_volume = self.kernel_num * (self.feature_size ** 2)

        # q
        self.q_mean = self._linear(self.feature_volume, self.z_size, relu=False)
        self.q_logvar = self._linear(self.feature_volume, self.z_size, relu=False)

        # projection
        self.project = self._linear(self.z_size, self.feature_volume, relu=False)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(self.kernel_num, self.kernel_num // 2),
            self._deconv(self.kernel_num // 2, self.kernel_num // 4),
            self._deconv(self.kernel_num // 4, self.channel_num),
            nn.Sigmoid()
        )

    def forward(self, x):
        # encode x
        encoded = self.encoder(x)

        # sample latent code z from q given x.
        mean, logvar = self.q(encoded)
        z = self.z(mean, logvar)
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )

        # reconstruct x from z
        x_reconstructed = self.decoder(z_projected)

        # return the parameters of distribution of q given x and the
        # reconstructed image.
        return (mean, logvar), x_reconstructed

    # ==============
    # VAE components
    # ==============

    def q(self, encoded):
        unrolled = encoded.view(-1, self.feature_volume)
        return self.q_mean(unrolled), self.q_logvar(unrolled)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = (
            Variable(torch.randn(std.size())).cuda() if self._is_on_cuda else
            Variable(torch.randn(std.size()))
        )
        return eps.mul(std).add_(mean)

    def reconstruction_loss(self, x_reconstructed, x):
        return F.mse_loss(x_reconstructed, x, reduction='sum')
        # return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    def kl_divergence_loss(self, mean, logvar):
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    # =====
    # Utils
    # =====

    @property
    def name(self):
        return (
            'VAE'
            '-{kernel_num}k'
            '-{channel_num}x{image_size}x{image_size}'
        ).format(
            kernel_num=self.kernel_num,
            image_size=self.image_size,
            channel_num=self.channel_num,
        )

    def sample(self, size):
        z = torch.randn(size, self.z_size).to(self.device)
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected)

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    # ======
    # Layers
    # ======

    def _conv(self, channel_size, kernel_num):
        return nn.Sequential(
            nn.Conv2d(
                channel_size, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _deconv(self, channel_num, kernel_num):
        return nn.Sequential(
            nn.ConvTranspose2d(
                channel_num, kernel_num,
                kernel_size=4, stride=2, padding=1,
            ),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(),
        )

    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.ReLU(),
        ) if relu else nn.Linear(in_size, out_size)

    # Lightning
    def training_step(self, batch, batch_idx):
        x, _ = batch
        (mean, logvar), x_reconstructed = self(x)
        reconstruction_loss = self.reconstruction_loss(x_reconstructed, x)
        kl_divergence_loss = self.kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + kl_divergence_loss

        self.log("train_reconstruction_loss", reconstruction_loss)
        self.log("train_kl_divergence_loss", kl_divergence_loss)
        self.log("train_total_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        (mean, logvar), x_reconstructed = self(x)
        reconstruction_loss = self.reconstruction_loss(x_reconstructed, x)
        kl_divergence_loss = self.kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + kl_divergence_loss

        self.log("val_reconstruction_loss", reconstruction_loss, on_epoch=True)
        self.log("val_kl_divergence_loss", kl_divergence_loss, on_epoch=True)
        self.log("val_total_loss", total_loss, on_epoch=True)

        if batch_idx == 0:
            log_image(self.logger, x_reconstructed, "val_reconstruction", self.current_epoch)
            z_projected = self.project(self.validation_sampling_z).view(
                -1, self.kernel_num,
                self.feature_size,
                self.feature_size,
            )
            sampled_images = self.decoder(z_projected)
            log_image(self.logger, sampled_images, "val_sampling", self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optim_lr,
                                     weight_decay=self.hparams.optim_weight_decay)
        return optimizer

