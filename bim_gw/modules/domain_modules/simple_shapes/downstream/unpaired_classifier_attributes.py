import torch.nn.functional as F
import torch.optim
import torchmetrics
from pytorch_lightning import LightningModule
from torch import nn

from bim_gw.modules.gw import split_domains_available_domains
from bim_gw.utils.text_composer.composer import composer
from bim_gw.utils.text_composer.utils import inspect_all_choices


class UnpairedClassifierAttributes(LightningModule):
    def __init__(self, global_workspace, optimizer_lr=1e-3, optimizer_weight_decay=1e-5):
        super(UnpairedClassifierAttributes, self).__init__()
        self.save_hyperparameters(ignore=["global_workspace"])

        self.global_workspace = global_workspace
        self.global_workspace.eval().freeze()

        self.text_composer = composer
        self.composer_inspection = inspect_all_choices(composer)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.regressor = nn.Sequential(
            nn.Linear(global_workspace.z_size, global_workspace.z_size),
            nn.ReLU(),
            nn.Linear(global_workspace.z_size, global_workspace.z_size // 2),
            nn.ReLU(),
            nn.Linear(global_workspace.z_size // 2, 1)
        )

    def step(self, batch, mode="train"):
        available_domains, domains = split_domains_available_domains(batch)
        latents = self.global_workspace.encode_uni_modal(domains)
        state = self.global_workspace.project(latents, keep_domains=['attr'])
        prediction = self.regressor(state)
        loss = F.mse_loss(prediction, latents['attr'][1][:, -1].unsqueeze(-1))
        bs = available_domains['attr'].shape[0]
        self.log(f"{mode}/loss", loss, logger=True, on_epoch=(mode != "train"), batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, mode="val")
        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.hparams.optimizer_lr, weight_decay=self.hparams.optimizer_weight_decay)
