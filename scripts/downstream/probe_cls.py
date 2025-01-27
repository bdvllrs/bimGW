import os
from typing import cast

import torch
import torch.nn.functional as F
import torch.optim
import torchmetrics
from omegaconf import OmegaConf
from pytorch_lightning import Callback, LightningModule
from torch import nn

from bim_gw.datasets import load_dataset
from bim_gw.datasets.simple_shapes.data_modules import SimpleShapesDataModule
from bim_gw.modules import GlobalWorkspace
from bim_gw.utils import get_args
from bim_gw.utils.scripts import get_domains, get_trainer
from bim_gw.utils.utils import get_checkpoint_path


class LinearProbes(LightningModule):
    def __init__(
        self,
        global_workspace: GlobalWorkspace,
        input_size_v: int,
        input_size_t: int,
        output_size: int,
        optimizer_lr: float = 1e-3,
        optimizer_weight_decay: float = 1e-5,
    ):
        super(LinearProbes, self).__init__()
        self.save_hyperparameters(ignore=["global_workspace"])

        self.global_workspace = global_workspace
        self.global_workspace.eval().freeze()
        self.output_size = output_size
        self.optimizer_lr = optimizer_lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.projection_gw_v = nn.Linear(
            global_workspace.z_size, self.output_size
        )
        self.projection_gw_t = nn.Linear(
            global_workspace.z_size, self.output_size
        )
        self.projection_v = nn.Linear(input_size_v, self.output_size)
        self.projection_t = nn.Linear(input_size_t, self.output_size)

        self.val_total = 0
        self.val_correct_gw_v = 0
        self.val_correct_gw_t = 0
        self.val_correct_v = 0
        self.val_correct_t = 0
        self.test_total = 0
        self.test_correct_gw_v = 0
        self.test_correct_gw_t = 0
        self.test_correct_v = 0
        self.test_correct_t = 0

    def v_prediction(self, latents) -> torch.Tensor:
        return self.projection_v(latents["v"].sub_parts["z_img"])

    def t_prediction(self, latents) -> torch.Tensor:
        return self.projection_t(latents["t"].sub_parts["z"])

    def gw_v_prediction(self, latents) -> torch.Tensor:
        state_v = gw.project({"v": latents["v"].sub_parts}, ["v"])
        return self.projection_gw_v(state_v)

    def gw_t_prediction(self, latents) -> torch.Tensor:
        state_t = gw.project({"t": latents["t"].sub_parts}, ["t"])
        return self.projection_gw_t(state_t)

    def step(self, batch, mode="train"):
        target = batch["attr"].sub_parts["cls"]
        batch_size = batch["attr"].sub_parts["cls"].size(0)

        latents = self.global_workspace.domains.encode(batch)

        gw_v_pred = self.gw_v_prediction(latents)
        loss_gw_v = F.cross_entropy(gw_v_pred, target)
        self.log(f"{mode}_loss_gw_v", loss_gw_v, batch_size=batch_size)

        gw_t_pred = self.gw_t_prediction(latents)
        loss_gw_t = F.cross_entropy(gw_t_pred, target)
        self.log(f"{mode}_loss_gw_t", loss_gw_t, batch_size=batch_size)

        v_pred = self.v_prediction(latents)
        loss_v = F.cross_entropy(v_pred, target)
        self.log(f"{mode}_loss_v", loss_v, batch_size=batch_size)

        t_pred = self.t_prediction(latents)
        loss_t = F.cross_entropy(t_pred, target)
        self.log(f"{mode}_loss_t", loss_t, batch_size=batch_size)

        loss = loss_gw_v + loss_gw_t + loss_v + loss_t
        self.log(f"{mode}_loss", loss, batch_size=batch_size)

        if mode == "val":
            self.val_total += batch_size
            self.val_correct_gw_v += (
                torch.argmax(gw_v_pred, -1) == target
            ).sum()
            self.val_correct_gw_t += (
                torch.argmax(gw_t_pred, -1) == target
            ).sum()
            self.val_correct_t += (torch.argmax(t_pred, -1) == target).sum()
            self.val_correct_v += (torch.argmax(v_pred, -1) == target).sum()
        if mode == "test":
            self.test_total += batch_size
            self.test_correct_gw_v += (
                torch.argmax(gw_v_pred, -1) == target
            ).sum()
            self.test_correct_gw_t += (
                torch.argmax(gw_t_pred, -1) == target
            ).sum()
            self.test_correct_t += (torch.argmax(t_pred, -1) == target).sum()
            self.test_correct_v += (torch.argmax(v_pred, -1) == target).sum()
        return loss

    def training_step(self, batch, _):
        loss = self.step(batch, mode="train")
        return loss

    def validation_step(self, batch, _):
        loss = self.step(batch, mode="val")
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_total = 0
        self.val_correct_gw_v = 0
        self.val_correct_gw_t = 0
        self.val_correct_v = 0
        self.val_correct_t = 0

    def on_validation_epoch_end(self) -> None:
        acc_v = self.val_correct_v / self.val_total
        acc_t = self.val_correct_t / self.val_total
        acc_gw_v = self.val_correct_gw_v / self.val_total
        acc_gw_t = self.val_correct_gw_t / self.val_total
        self.log("val_acc_v", acc_v)
        self.log("val_acc_t", acc_t)
        self.log("val_acc_gw_v", acc_gw_v)
        self.log("val_acc_gw_t", acc_gw_t)

    def test_step(self, batch, _):
        loss = self.step(batch, mode="test")
        return loss

    def on_test_epoch_start(self) -> None:
        self.test_total = 0
        self.test_correct_gw_v = 0
        self.test_correct_gw_t = 0
        self.test_correct_v = 0
        self.test_correct_t = 0

    def on_test_epoch_end(self) -> None:
        acc_v = self.test_correct_v / self.test_total
        acc_t = self.test_correct_t / self.test_total
        acc_gw_v = self.test_correct_gw_v / self.test_total
        acc_gw_t = self.test_correct_gw_t / self.test_total
        self.log("test_acc_v", acc_v)
        self.log("test_acc_t", acc_t)
        self.log("test_acc_gw_v", acc_gw_v)
        self.log("test_acc_gw_t", acc_gw_t)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(
            params,
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )


if __name__ == "__main__":
    args = get_args(debug=bool(int(os.getenv("DEBUG", 0))))

    args.global_workspace.selected_domains = OmegaConf.create(
        ["v", "attr", "t"]
    )

    data: SimpleShapesDataModule = load_dataset(
        args, args.global_workspace, add_unimodal=False
    )
    data.prepare_data()
    data.setup(stage="fit")

    for model in args.downstream.clip_cls.checkpoints:
        print(model)
        checkpoint_path = get_checkpoint_path(model)
        domain_mods = get_domains(args, data.img_size)
        gw = cast(
            GlobalWorkspace,
            GlobalWorkspace.load_from_checkpoint(
                str(checkpoint_path),
                domain_mods=domain_mods,
                optim_unsupervised_losses_after=args.global_workspace.optim.unsupervised_losses_after_n_epochs,
                domain_examples=data.domain_examples,
                strict=False,
            ),
        )
        gw.eval().freeze()

        num_classes = 3

        probe_gw_v = LinearProbes(
            gw,
            domain_mods["v"].domain_specs.output_dims["z_img"],
            domain_mods["t"].domain_specs.output_dims["z"],
            num_classes,
        )

        trainer = get_trainer(
            "probe_cls",
            args,
            probe_gw_v,
            monitor_loss="val_loss",
            trainer_args={
                # "val_check_interval": args.global_workspace.prop_labelled_images
            },
        )

        trainer.fit(probe_gw_v, data)
        trainer.validate(probe_gw_v, data)
        trainer.test(probe_gw_v, data)
