import torch.nn.functional as F
import torch.optim
import torchmetrics
from pytorch_lightning import LightningModule
from torch import nn

from bim_gw.modules.gw import split_domains_available_domains
from bim_gw.utils.text_composer.composer import composer
from bim_gw.utils.text_composer.utils import inspect_all_choices


class UnpairedClassifierText(LightningModule):
    def __init__(
        self,
        global_workspace,
        hidden_size=64,
        optimizer_lr=1e-3,
        optimizer_weight_decay=1e-5,
    ):
        super(UnpairedClassifierText, self).__init__()
        self.save_hyperparameters(ignore=["global_workspace"])

        self.global_workspace = global_workspace
        self.global_workspace.eval().freeze()

        self.text_composer = composer
        self.composer_inspection = inspect_all_choices(composer)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

        self.hidden_size = hidden_size

        self.projection = nn.Sequential(
            nn.Linear(global_workspace.z_size, global_workspace.z_size),
            nn.ReLU(),
            nn.Linear(global_workspace.z_size, global_workspace.z_size // 2),
            nn.ReLU(),
            nn.Linear(global_workspace.z_size // 2, self.hidden_size),
        )
        self.grammar_classifiers = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size, n_outputs),
                )  # predict sentence structure
                for name, n_outputs in self.composer_inspection.items()
            }
        )

        self.grammar_train_acc = nn.ModuleDict(
            {
                name: torchmetrics.Accuracy()
                for name in self.composer_inspection.keys()
            }
        )
        self.grammar_val_acc = nn.ModuleDict(
            {
                name: torchmetrics.Accuracy()
                for name in self.composer_inspection.keys()
            }
        )

    def step(self, batch, mode="train"):
        available_domains, domains = split_domains_available_domains(batch)
        latents = self.global_workspace.domains.encode(domains)
        state = self.global_workspace.project(latents, keep_domains=["t"])
        z = self.projection(state)
        total_loss = 0
        bs = available_domains["t"].shape[0]
        for (
            grammar_type,
            grammar_classifier,
        ) in self.grammar_classifiers.items():
            prediction = grammar_classifier(z)
            loss_grammar = F.cross_entropy(
                prediction, domains["t"][2][grammar_type]
            )
            total_loss += loss_grammar
            acc_fn = (
                self.grammar_train_acc[grammar_type]
                if mode == "train"
                else self.grammar_val_acc[grammar_type]
            )
            res = acc_fn(prediction.softmax(-1), domains["t"][2][grammar_type])
            self.log(
                f"{mode}/loss_{grammar_type}",
                loss_grammar,
                logger=True,
                on_epoch=(mode != "train"),
                batch_size=bs,
            )
            self.log(
                f"{mode}_{grammar_type}_acc", res, on_epoch=(mode != "train")
            )

        self.log(
            f"{mode}/total_loss",
            total_loss,
            logger=True,
            on_epoch=(mode != "train"),
            batch_size=bs,
        )

        return total_loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, mode="val")
        return loss

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(
            params,
            lr=self.hparams.optimizer_lr,
            weight_decay=self.hparams.optimizer_weight_decay,
        )
