import torch.optim
import torchmetrics
from pytorch_lightning import LightningModule
from torch import nn


class OddClassifier(LightningModule):
    def __init__(self, encoder, z_size, optimizer_lr=1e-3, optimizer_weight_decay=1e-5):
        super(OddClassifier, self).__init__()
        self.save_hyperparameters(ignore=["encoder"])
        self.encoder = encoder
        self.z_size = z_size

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.classifier = nn.Sequential(
            nn.Linear(self.z_size * 3, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def step(self, batch, mode="train"):
        latents = torch.cat([
            self.encoder([batch[0]]),
            self.encoder([batch[1]]),
            self.encoder([batch[2]]),
        ], dim=1)

        prediction = self.classifier(latents)
        loss = torch.nn.functional.cross_entropy(prediction, batch[3])
        self.log(f"{mode}_loss", loss)
        acc_fn = self.train_acc if mode == "train" else self.val_acc
        acc_fn(prediction.softmax(-1), batch[3])
        self.log(f"{mode}_acc", acc_fn, on_epoch=(mode=="val"))
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
