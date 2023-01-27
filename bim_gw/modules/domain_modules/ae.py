from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from bim_gw.modules.domain_modules.domain_module import DomainModule
from bim_gw.utils.utils import log_image


class AE(DomainModule):
    def __init__(self, image_size: int, channel_num: int, ae_size: int, z_size: int,
                 n_validation_examples: int = 32,
                 optim_lr: float = 3e-4, optim_weight_decay: float = 1e-5,
                 scheduler_step: int = 20, scheduler_gamma: float = 0.5,
                 validation_reconstruction_images: Optional[torch.Tensor] = None):
        # configurations
        super().__init__()
        self.save_hyperparameters(ignore=["validation_reconstruction_images"])

        self.image_size = image_size
        assert channel_num in [1, 3]
        self.channel_num = channel_num
        self.ae_size = ae_size
        self.z_size = z_size

        self.output_dims = [self.z_size]
        self.decoder_activation_fn = [
            None
        ]
        self.losses = [F.mse_loss]
        # val sampling
        if validation_reconstruction_images is not None:
            self.register_buffer("validation_reconstruction_images", validation_reconstruction_images)
        else:
            self.validation_reconstruction_images = None
        # self.log_sigma = nn.Parameter(torch.tensor(0.), requires_grad=True)

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

        self.decoder = CDecoderV2(channel_num, image_size, ae_size=ae_size, z_size=self.z_size, batchnorm=True)

    def encode(self, x: torch.Tensor):
        return self.q_mean(self.encoder(x[0])),

    def decode(self, z: torch.Tensor):
        return self.decoder(z[0]),

    def forward(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        z = self.encode(x)

        # reconstruct x from z
        x_reconstructed = self.decoder(z)

        return z, x_reconstructed

    def reconstruction_loss(self, x_reconstructed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        assert x_reconstructed.size() == x.size()
        return F.mse_loss(x_reconstructed, x)

    def step(self, batch, mode="train"):
        x = batch[0]["v"][1]
        z, x_reconstructed = self(x)
        reconstruction_loss = self.reconstruction_loss(x_reconstructed, x)
        total_loss = reconstruction_loss

        # self.log(f"{mode}_mse_grads", grad_norm(reconstruction_loss, self.parameters(), retain_graph=True, allow_unused=True))
        # self.log(f"{mode}_kl_grads", grad_norm(kl_divergence_loss, self.parameters(), retain_graph=True, allow_unused=True))

        self.log(f"{mode}_reconstruction_loss", reconstruction_loss, logger=True, on_epoch=(mode != "train"))
        self.log(f"{mode}_total_loss", total_loss, on_epoch=(mode != "train"))

        return total_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, mode="test")

    def epoch_end(self, outputs, mode="val"):
        if self.validation_reconstruction_images is not None:
            for logger in self.loggers:
                x = self.validation_reconstruction_images
                _, x_reconstructed = self(x)

                if self.current_epoch == 0:
                    log_image(logger, x[:self.hparams.n_validation_examples], f"{mode}_original_images")

                log_image(logger, x_reconstructed[:self.hparams.n_validation_examples], f"{mode}_reconstruction")

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

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, mode="val")

    def test_epoch_end(self, outputs):
        self.epoch_end(outputs, mode="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optim_lr,
                                     weight_decay=self.hparams.optim_weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.scheduler_step,
                                                    self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]

    def log_domain(self, logger, x, title, step=None):
        log_image(logger, x[0], title, step)


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
