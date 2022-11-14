import torch.optim
import torchmetrics
from pytorch_lightning import LightningModule
from torch import nn


class OddClassifier(LightningModule):
    def __init__(self, unimodal_encoders, encoders, z_size, optimizer_lr=1e-3, optimizer_weight_decay=1e-5):
        super(OddClassifier, self).__init__()
        self.save_hyperparameters(ignore=["encoder"])
        self.unimodal_encoders = nn.ModuleDict(unimodal_encoders)
        self.encoders = nn.ModuleDict(encoders)
        self.z_size = z_size

        for mod in unimodal_encoders.values():
            mod.freeze()  # insures that all modules are frozen

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.classifier = nn.Sequential(
            nn.Linear(self.z_size * 3, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def step(self, batch, mode="train"):
        latents = {
            name: torch.cat([
                    self.encoders[name](self.unimodal_encoders[name](batch[name][0])),
                    self.encoders[name](self.unimodal_encoders[name](batch[name][1])),
                    self.encoders[name](self.unimodal_encoders[name](batch[name][2])),
                ], dim=1)
            for name in self.unimodal_encoders.keys()
        }

        predictions = {
            name: self.classifier(latents[name])
            for name in self.unimodal_encoders.keys()
        }
        losses = {
            name: torch.nn.functional.cross_entropy(predictions[name], batch['label'])
            for name in self.unimodal_encoders.keys()
        }
        for name in losses.keys():
            self.log(f"{mode}_{name}_loss", losses[name])
            acc_fn = self.train_acc if mode == "train" else self.val_acc
            acc_fn(predictions[name].softmax(-1), batch['label'])
            self.log(f"{mode}_{name}_acc", acc_fn, on_epoch=(mode=="val"))
        self.log(f"{mode}_loss", losses["v"])
        return losses["v"]

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, mode="val")
        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.hparams.optimizer_lr, weight_decay=self.hparams.optimizer_weight_decay)
