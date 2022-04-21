from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from bim_gw.modules.workspace_module import WorkspaceModule
from bim_gw.utils.losses.mmd import mmd_loss
from bim_gw.utils.utils import log_image
from bim_gw.utils.vae import reparameterize, gaussian_nll, softclip


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

        self.layer1 = nn.Sequential(
            DeconvResNetBlock(512, 512, 3, 1, 0, True),
            DeconvResNetBlock(512, 256, 3, 2, 1, True),
        )
        self.layer2 = nn.Sequential(
            DeconvResNetBlock(256, 256, 3, 1, 0, True),
            DeconvResNetBlock(256, 128, 3, 2, 1, True),
        )
        self.layer3 = nn.Sequential(
            DeconvResNetBlock(128, 128, 3, 1, 0, True),
            DeconvResNetBlock(128, 64, 3, 2, 1, True),
        )
        self.layer4 = nn.Sequential(
            DeconvResNetBlock(64, 64, 3, 1, 0, False),
            DeconvResNetBlock(64, 64, 3, 1, 0, False),
        )

        self.unpool = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.ConvTranspose2d(64, 3, 7, 2, 3, output_padding=1, bias=False)
        self.linear = nn.Linear(self.z_dim, 512, bias=False)
        self.bn = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.linear(x)  # (B, 512)
        x = x.reshape(x.size(0), 512, 1, 1)  # (B, 512, 1, 1)
        out = self.bn(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.unpool(out)
        out = self.conv(out)
        return out


class VAE(WorkspaceModule):
    def __init__(self, image_size: int, channel_num: int, ae_size: int, z_size: int, beta: float = 1,
                 vae_type="beta",
                 n_validation_examples: int = 32,
                 optim_lr: float = 3e-4, optim_weight_decay: float = 1e-5,
                 scheduler_step: int = 20, scheduler_gamma: float = 0.5,
                 validation_reconstruction_images: Optional[torch.Tensor] = None,
                 n_FID_samples=1000,
                 mmd_loss_coef=0.,
                 kl_loss_coef=0.):
        # configurations
        super().__init__()
        self.save_hyperparameters(ignore=["validation_reconstruction_images"])

        self.image_size = image_size
        assert channel_num in [1, 3]
        assert beta >= 0. and optim_lr >= 0 and optim_weight_decay >= 0 and scheduler_step >= 0
        self.channel_num = channel_num
        self.ae_size = ae_size
        self.z_size = z_size
        self.beta = beta
        assert vae_type in ["beta", "sigma", "optimal_sigma"]
        self.vae_type = vae_type
        self.n_FID_samples = n_FID_samples

        self.output_dims = [1, self.z_size]
        self.decoder_activation_fn = [
            torch.sigmoid,
            None
        ]
        self.losses = [
            F.binary_cross_entropy,
            lambda x, y: (
                F.mse_loss(x, y) +
                mmd_loss_coef * mmd_loss(x, y) +
                kl_loss_coef * self.kl_divergence_loss(x.mean(0), x.var(0).log())
            )
        ]

        # val sampling
        self.register_buffer("validation_sampling_z", torch.randn(n_validation_examples, self.z_size))

        if validation_reconstruction_images is not None:
            self.register_buffer("validation_reconstruction_images", validation_reconstruction_images)
        else:
            self.validation_reconstruction_images = None

        if self.vae_type == "sigma":
            self.log_sigma = nn.Parameter(torch.tensor(0.), requires_grad=True)
        else:
            self.register_buffer("log_sigma", torch.tensor(0.))

        # self.encoder = torchvision.models.resnet18(False)
        # self.encoder.fc = nn.Identity()
        #
        # # q
        # self.q_mean = nn.Linear(512, self.z_size)
        # self.q_logvar = nn.Linear(512, self.z_size)
        #
        # # decoder
        # self.decoder = ResNetDecoder(z_size)
        self.encoder = CEncoderV2(channel_num, image_size, ae_size=ae_size, batchnorm=True)

        self.q_mean = nn.Linear(self.encoder.out_size, self.z_size)
        self.q_logvar = nn.Linear(self.encoder.out_size, self.z_size)

        self.decoder = CDecoderV2(channel_num, image_size, ae_size=ae_size, z_size=self.z_size, batchnorm=True)

    def encode_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(x)
        out = out.view(out.size(0), -1)

        mean_z = self.q_mean(out)
        var_z = self.q_logvar(out)
        return mean_z, var_z

    def encode(self, x: torch.Tensor):
        is_active, x = x
        mean_z, _ = self.encode_stats(x)

        # z = reparameterize(mean_z, var_z)
        return is_active, mean_z

    def decode(self, z: torch.Tensor):
        is_active, z = z
        return is_active, self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        mean, logvar = self.encode_stats(x)
        z = reparameterize(mean, logvar)

        # reconstruct x from z
        x_reconstructed = self.decoder(z)

        return (mean, logvar), x_reconstructed

    def reconstruction_loss(self, x_reconstructed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        assert x_reconstructed.size() == x.size()
        if self.vae_type == "optimal_sigma":
            log_sigma = ((x - x_reconstructed) ** 2).mean([0, 1, 2, 3], keepdim=True).sqrt().log()
            log_sigma = softclip(log_sigma, -6)
            self.log_sigma = log_sigma.squeeze()
        loss = gaussian_nll(x_reconstructed, self.log_sigma, x).sum()
        return loss

    def kl_divergence_loss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return kl

    def sample(self, size: int) -> torch.Tensor:
        z = torch.randn(size, self.z_size).to(self.device)
        return self.decoder(z)

    def generate(self, samples):
        return self.decode(samples)

    def step(self, batch, mode="train"):
        x = batch[0]["v"][1]

        (mean, logvar), x_reconstructed = self(x)
        reconstruction_loss = self.reconstruction_loss(x_reconstructed, x)
        kl_divergence_loss = self.kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + self.beta * kl_divergence_loss

        self.log(f"{mode}_reconstruction_loss", reconstruction_loss, logger=True, on_epoch=(mode == "val"))
        self.log(f"{mode}_kl_divergence_loss", kl_divergence_loss, logger=True, on_epoch=(mode == "val"))
        self.log(f"{mode}_total_loss", total_loss, on_epoch=(mode == "val"))
        if mode == "train":
            self.log("log_sigma", self.log_sigma, logger=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode="val")

    def validation_epoch_end(self, outputs):
        if self.validation_reconstruction_images is not None:
            for logger in self.loggers:
                x = self.validation_reconstruction_images
                _, x_reconstructed = self(x)

                if self.current_epoch == 0:
                    log_image(logger, x[:self.hparams.n_validation_examples], "val_original_images")

                log_image(logger, x_reconstructed[:self.hparams.n_validation_examples], "val_reconstruction")
                sampled_images = self.decoder(self.validation_sampling_z)
                log_image(logger, sampled_images, "val_sampling")

                # # FID
                # fid, mse = compute_FID(
                #     self.trainer.datamodule.inception_stats_path_train,
                #     self.trainer.datamodule.val_dataloader()[0],
                #     self, self.z_size, [self.image_size, self.image_size],
                #     self.device, self.n_FID_samples
                # )
                # self.log("val_fid", fid)
                # # self.print("FID: ", fid)
                # self.log("val_mse", mse)

                #
                # stat_train = np.load(self.trainer.datamodule.inception_stats_path_train, allow_pickle=True).item()
                # mu_dataset_train = stat_train['mu']
                # sigma_dataset_train = stat_train['sigma']
                #
                # stat_test = np.load(self.trainer.datamodule.inception_stats_path_val, allow_pickle=True).item()
                # mu_dataset_test = stat_test['mu']
                # sigma_dataset_test = stat_test['sigma']
                #
                # fid_value = calculate_frechet_distance(mu_dataset_train, sigma_dataset_train, mu_dataset_test, sigma_dataset_test)
                # self.print("FID test: ", fid_value)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optim_lr,
                                     weight_decay=self.hparams.optim_weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.scheduler_step,
                                                    self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]

    def log_domain(self, logger, x, title, max_examples=None):
        is_active, x = x
        log_image(logger, x[:max_examples], title)


class CEncoderV2(nn.Module):
    def __init__(self, num_channels, input_size, ae_size=1028, batchnorm=False):
        super().__init__()

        if input_size >= 32:
            sizes = [ae_size // (2 ** i) for i in reversed(range(4))]  # 1 2 4 8 # 32 64 128 256
        else:
            sizes = [32, 64, 128, 256]

        if input_size == 64:
            # kernel_size = 4
            # padding = 1
            kernel_size = 5
            padding = 2
        else:
            kernel_size = 4
            padding = 1

        if ae_size is not None:
            sizes[-1] = ae_size

        self.layers = nn.Sequential(
            nn.Conv2d(num_channels, sizes[0], kernel_size=kernel_size, stride=2, padding=padding, bias=not batchnorm),
            nn.BatchNorm2d(sizes[0]) if batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(sizes[0], sizes[1], kernel_size=kernel_size, stride=2, padding=padding, bias=not batchnorm),
            nn.BatchNorm2d(sizes[1]) if batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(sizes[1], sizes[2], kernel_size=kernel_size, stride=2, padding=padding, bias=not batchnorm),
            nn.BatchNorm2d(sizes[2]) if batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(sizes[2], sizes[3], kernel_size=kernel_size, stride=2, padding=padding, bias=not batchnorm),
            nn.BatchNorm2d(sizes[3]) if batchnorm else nn.Identity(),
            nn.ReLU(),
        )

        self.out_size = sizes[3] * 2 * 2 * (input_size // 32) * (input_size // 32)

    def forward(self, x):
        # x = self.layers(x).view(x.size(0), -1)
        # print(x.shape)
        # return x
        return self.layers(x).view(x.size(0), -1)


class CDecoderV2(nn.Module):
    def __init__(self, num_channels, input_size, z_size, ae_size=1028, batchnorm=False):
        super().__init__()

        if input_size <= 32:
            if input_size >= 32:
                sizes = [ae_size // (2 ** i) for i in reversed(range(3))]  # 1 2 4 8 # 32 64 128 256
            else:
                sizes = [64, 128, 256]

            if ae_size is not None:
                sizes[-1] = ae_size

            kernel_size = 4
            padding = 1

            self.layers = nn.Sequential(
                nn.ConvTranspose2d(z_size, sizes[2], kernel_size=8, stride=1, bias=not batchnorm),
                nn.BatchNorm2d(sizes[2]) if batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.ConvTranspose2d(sizes[2], sizes[1], kernel_size=kernel_size, stride=2, padding=padding,
                                   bias=not batchnorm),
                nn.BatchNorm2d(sizes[1]) if batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.ConvTranspose2d(sizes[1], sizes[0], kernel_size=kernel_size, stride=2, padding=padding,
                                   bias=not batchnorm),
                nn.BatchNorm2d(sizes[0]) if batchnorm else nn.Identity(),
                nn.ReLU(),
            )
            out_padding_layer = nn.ZeroPad2d((0, 1, 0, 1))
            final_padding = 1

            out_size = sizes[0]
        else:
            ae_size = 1024 if ae_size is None else ae_size

            sizes = [ae_size // (2 ** i) for i in reversed(range(4))]

            kernel_size = 5
            padding = 2

            self.layers = nn.Sequential(
                nn.ConvTranspose2d(z_size, sizes[3], kernel_size=8, stride=1, bias=not batchnorm),
                nn.BatchNorm2d(sizes[3]) if batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.ConvTranspose2d(sizes[3], sizes[2], kernel_size=kernel_size, stride=2, padding=padding,
                                   output_padding=1, bias=not batchnorm),
                nn.BatchNorm2d(sizes[2]) if batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.ConvTranspose2d(sizes[2], sizes[1], kernel_size=kernel_size, stride=2, padding=padding,
                                   output_padding=1, bias=not batchnorm),
                nn.BatchNorm2d(sizes[1]) if batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.ConvTranspose2d(sizes[1], sizes[0], kernel_size=kernel_size, stride=2, padding=padding,
                                   output_padding=1, bias=not batchnorm),
                nn.BatchNorm2d(sizes[0]) if batchnorm else nn.Identity(),
                nn.ReLU(),
            )
            out_padding_layer = nn.Identity()
            # final_padding = 2

            out_size = sizes[0]

        self.out_layer = nn.Sequential(
            out_padding_layer,
            nn.Conv2d(sizes[0], num_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.Sigmoid()
        )

        self.output_fun = self.out_layer

    def forward(self, z):
        return self.output_fun(self.layers(z[:, :, None, None]))
