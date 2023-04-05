from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule
from torch import nn

from bim_gw.datasets.domain import DomainItems
from bim_gw.modules.domain_interface import DomainInterface
from bim_gw.modules.domain_modules import DomainModule
from bim_gw.utils.types import SchedulerInterval, SchedulerMode
from bim_gw.utils.utils import log_if_save_last_images, log_if_save_last_tables


def check_domains_eq(domains_ori, domains_comp):
    for o, r in zip(domains_ori.values(), domains_comp.values()):
        assert torch.eq(o[0], r[0]).all()
        if o[1] is not None:
            assert torch.eq(o[1], r[1]).all()


def split_domains_available_domains(domains):
    indicators = {domain_name: domain[0].to(torch.bool) for domain_name, domain
                  in domains.items()}
    domains = {domain_name: domain[1:] for domain_name, domain in
               domains.items()}
    return indicators, domains


class GlobalWorkspace(LightningModule):
    def __init__(
        self, domain_mods: Dict[str, DomainModule], z_size, hidden_size,
        n_layers_encoder,
        n_layers_decoder, n_layers_decoder_head,
        n_classes=1000,
        loss_coef_demi_cycles=1., loss_coef_cycles=1.,
        loss_coef_translation=1., loss_coef_cosine=0.,
        loss_coef_contrastive=0., optim_lr=3e-4, optim_weight_decay=1e-5,
        scheduler_mode: SchedulerMode = SchedulerMode.fixed,
        scheduler_interval: SchedulerInterval = SchedulerInterval.epoch,
        scheduler_step=20, scheduler_gamma=0.5,
        loss_schedules=None,
        monitor_grad_norms: bool = False,
        remove_sync_domains=None,
        save_only_last_images=False,
    ):
        super(GlobalWorkspace, self).__init__()

        self.loss_coef_demi_cycles = loss_coef_demi_cycles
        self.loss_coef_cycles = loss_coef_cycles
        self.loss_coef_translation = loss_coef_translation
        self.loss_coef_contrastive = loss_coef_contrastive
        self.loss_coef_cosine = loss_coef_cosine
        self.loss_schedules = {}
        if loss_schedules is not None:
            self.loss_schedules = loss_schedules
        self.remove_sync_domains = []
        if remove_sync_domains is not None:
            self.remove_sync_domains = remove_sync_domains
        self.remove_sync_domains_names = [
            f"{n1}-{n2}" for n1, n2 in self.remove_sync_domains
        ]
        self.remove_sync_domains_names += [
            f"{n2}-{n1}" for n1, n2 in self.remove_sync_domains
        ]

        self.z_size = z_size
        self.hidden_size = hidden_size
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_layers_decoder_head = n_layers_decoder_head
        self.monitor_grad_norms = monitor_grad_norms
        self.save_only_last_images = save_only_last_images

        self.domains = DomainInterface(domain_mods)

        self.hparams["contrastive_logit_scale"] = {
            f"{n1}-{n2}": torch.ones([]) * np.log(1 / 0.07)
            for i, n1 in enumerate(self.domains.names)
            for j, n2 in enumerate(self.domains.names)
            if i < j
        }

        # Define encoders for translation
        encoders = {}
        decoders = {}
        for item, domain_specs in self.domains.get_specs():
            encoder_class = domain_specs.workspace_encoder_cls
            decoder_class = domain_specs.workspace_decoder_cls
            encoders[item] = encoder_class(
                domain_specs,
                self.hidden_size['encoder'][item],
                self.z_size, self.n_layers_encoder[item]
            )
            decoders[item] = decoder_class(
                domain_specs,
                self.z_size, self.hidden_size['decoder'][item],
                self.n_layers_decoder[item],
                self.n_layers_decoder_head[item],
            )
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)

        # Accuracies
        train_accuracy_metrics = []
        val_in_dist_accuracy_metrics = []
        val_ood_accuracy_metrics = []
        self.accuracy_metrics_order = []
        for domain_name, domain_specs in self.domains.get_specs():
            if domain_specs.requires_acc_computation:
                for domain_name_start in self.domains.names:
                    if domain_name_start != domain_name:
                        train_accuracy_metrics.append(torchmetrics.Accuracy())
                        val_in_dist_accuracy_metrics.append(
                            torchmetrics.Accuracy()
                        )
                        val_ood_accuracy_metrics.append(
                            torchmetrics.Accuracy()
                        )
                        self.accuracy_metrics_order.append(
                            (domain_name_start, domain_name)
                        )

        self.train_accuracy_metrics = nn.ModuleList(train_accuracy_metrics)
        self.val_in_dist_accuracy_metrics = nn.ModuleList(
            val_in_dist_accuracy_metrics
        )
        self.val_ood_accuracy_metrics = nn.ModuleList(val_ood_accuracy_metrics)

        self.domain_examples = None

        self.rotation_error_val = []

        self.save_hyperparameters(
            ignore=["domain_mods", "domain_examples", "loss_schedules"]
        )
        print("Global Workspace instantiated.")

    def encode(self, x, domain_name, add_tanh=True):
        pre_act = self.encoders[domain_name](x)
        if add_tanh:
            return torch.tanh(pre_act)
        return pre_act

    def decode(self, z, domain_name):
        return self.decoders[domain_name](z)

    def project(
        self, latents: Dict[str, Dict[str, torch.Tensor]],
        keep_domains
    ):
        pre_act = 0
        assert len(keep_domains), "Must project at least one domain"
        for domain in keep_domains:
            pre_act += self.encode(latents[domain], domain, add_tanh=False)
        return torch.tanh(pre_act)

    def predict(self, state):
        return {
            domain_name: self.decode(state, domain_name)
            for domain_name in self.domains.names
        }

    def forward(self, domains):
        latents = self.domains.encode(domains)
        return self.project(latents, list(latents.keys()))

    def translate(self, x, domain_name_start, domain_name_target):
        """
        Translates x from domain1 to domain2
        """
        z = self.encode(x, domain_name_start)
        return self.decode(z, domain_name_target)

    def demi_cycle(self, x, domain_inter):
        return self.translate(x, domain_inter, domain_inter)

    def cycle(self, x, domain_name_start, domain_name_inter):
        z = self.translate(x, domain_name_start, domain_name_inter)
        return self.translate(z, domain_name_inter, domain_name_start)

    def schedule_loss_coef_step(self):
        for loss_name in ["demi_cycles", "cycles", "translation", "cosine"]:
            if loss_name in self.loss_schedules:
                coef = getattr(self, f"loss_coef_{loss_name}")
                epoch_rest = self.current_epoch
                epoch_rest %= self.loss_schedules[loss_name]["step"] == 0
                if (
                        self.current_epoch != 0
                        and epoch_rest
                ):
                    setattr(
                        self, f"loss_coef_{loss_name}",
                        coef * self.loss_schedules[loss_name]["gamma"]
                    )

    def contrastive_loss(self, states):
        losses = []
        indiv_losses = {}
        for domain_name in states.keys():
            if domain_name in self.hparams.contrastive_logit_scale.keys():
                domain_name_1, domain_name_2 = domain_name.split("-")
                latent_domain_1 = states[domain_name]
                latent_domain_2 = states[f"{domain_name_2}-{domain_name_1}"]
                # project domains into one another
                latent_features_d1 = latent_domain_1 / latent_domain_1.norm(
                    dim=1, keepdim=True
                )
                latent_features_d2 = latent_domain_2 / latent_domain_2.norm(
                    dim=1, keepdim=True
                )
                logit_scale = self.hparams.contrastive_logit_scale[
                    f"{domain_name_1}-{domain_name_2}"].exp()
                logits = (
                        logit_scale * latent_features_d1
                        @ latent_features_d2.t()
                )
                labels = torch.arange(latent_domain_1.size(0)).to(
                    logits.device
                )
                loss_d1 = F.cross_entropy(logits, labels)
                loss_d2 = F.cross_entropy(logits.t(), labels)
                loss = .5 * (loss_d1 + loss_d2)
                indiv_losses[
                    f"contrastive_s_{domain_name_1}-s_{domain_name_2}"] = loss
                losses.append(loss)
        if len(losses):
            indiv_losses["contrastive"] = torch.stack(losses, dim=0).mean()
            return indiv_losses
        return {"contrastive": torch.tensor(0.).to(self.device)}

    def loss(self, predictions, targets, prefix=""):
        losses = []
        indiv_losses = {}
        for domain_name in predictions.keys():
            loss_domain = domain_name
            if "-" in domain_name:
                loss_domain = domain_name.split("-")[0]
            prediction, target = predictions[domain_name], targets[domain_name]
            (domain_total,
             domain_indiv_losses) = self.domains[loss_domain].loss(
                prediction, target
            )
            indiv_losses.update(
                {f"{prefix}/domain_{domain_name}_{k}": v for k, v in
                 domain_indiv_losses.items()}
            )
            losses.append(domain_total)
            indiv_losses[f"{prefix}/domain_{domain_name}"] = domain_total
        if len(losses):
            indiv_losses[prefix] = torch.stack(losses, dim=0).mean()
            return indiv_losses
        return {prefix: torch.tensor(0.).to(self.device)}

    def step(
        self, latents: Dict[str, DomainItems], mode: str = "val",
        prefix: str = ""
    ):
        latent_demi_cycle_predictions = {}
        latent_demi_cycle_target = {}
        latent_cycle_predictions = {}
        latent_cycle_target = {}
        latent_translation_predictions = {}
        latent_translation_target = {}
        latent_translation_predictions_2 = {}
        latent_translation_target_2 = {}
        states = {}

        latents_sub_parts = {
            domain_name: latent.sub_parts
            for domain_name, latent in latents.items()
        }
        for domain_name, latent in latents.items():
            available_items = latent.available_masks
            if available_items.any():
                # Demi-cycles
                state = self.project(latents_sub_parts, [domain_name])
                predictions = self.predict(state)
                latent_demi_cycle_predictions[f"{domain_name}-u"] = {
                    k: predictions[domain_name][k][available_items]
                    for k in predictions[domain_name].keys()
                }
                latent_demi_cycle_target[f"{domain_name}-u"] = {
                    k: latent[k][available_items]
                    for k in latent.keys()
                }
                for domain_name_target, latent_target in latents.items():
                    if domain_name_target != domain_name:
                        # Cycles
                        cycle_state = self.project(
                            self.domains.adapt(predictions),
                            [domain_name_target]
                        )
                        cycle_prediction = self.decode(
                            cycle_state, domain_name
                        )
                        latent_cycle_predictions[
                            f"{domain_name}-{domain_name_target}"] = {
                            k: cycle_prediction[k][available_items]
                            for k in cycle_prediction.keys()
                        }
                        latent_cycle_target[
                            f"{domain_name}-{domain_name_target}"] = {
                            k: latent[k][available_items]
                            for k in latent.keys()
                        }
                        # Translation
                        mask = torch.logical_and(
                            available_items,
                            latents[domain_name_target].available_masks
                        )
                        if not mask.any():
                            continue

                        states[f"{domain_name}-{domain_name_target}"] = (
                            state[mask]
                        )
                        pred = latent_translation_predictions
                        target = latent_translation_target
                        if (
                                f"{domain_name}-{domain_name_target}" in
                                self.remove_sync_domains_names
                        ):
                            pred = latent_translation_predictions_2
                            target = latent_translation_target_2

                        pred[f"{domain_name_target}-{domain_name}"] = {
                            k: predictions[domain_name_target][k][mask]
                            for k in predictions[domain_name_target].keys()
                        }
                        target[f"{domain_name_target}-{domain_name}"] = {
                            k: latent_target[k][mask]
                            for k in latent_target.keys()
                        }

        demi_cycle_losses = self.loss(
            latent_demi_cycle_predictions, latent_demi_cycle_target,
            prefix="demi_cycles"
        )
        cycle_losses = self.loss(
            latent_cycle_predictions, latent_cycle_target, prefix="cycles"
        )
        contrastive_losses = self.contrastive_loss(states)

        translation_losses = self.loss(
            latent_translation_predictions, latent_translation_target,
            prefix="translation"
        )
        translation_losses_2 = {}
        if mode == "val":
            translation_losses_2 = self.loss(
                latent_translation_predictions_2, latent_translation_target_2,
                prefix="translation-non-op"
            )

        losses = {
            **demi_cycle_losses, **cycle_losses, **translation_losses,
            **contrastive_losses,
            **translation_losses_2
        }
        loss_names = ["demi_cycles", "cycles", "translation", "contrastive"]
        losses["total"] = torch.stack(
            [self.hparams[f"loss_coef_{loss_name}"] * losses[loss_name]
             for loss_name in loss_names], dim=0
        ).sum()
        losses["total_no_coefs"] = torch.stack(
            [losses[loss_name] for loss_name in loss_names], dim=0
        ).sum()

        batch_size = latents[list(latents.keys())[0]].available_masks.size(0)
        for name, loss in losses.items():
            loss_name = f"{mode}/{prefix}{name}_loss"
            if mode == "val" and self.trainer.global_step == 0:
                for logger in self.loggers:
                    logger.set_summary(loss_name, "min")
            self.log(
                loss_name, loss, logger=True,
                add_dataloader_idx=False, batch_size=batch_size
            )

        if mode == "train":
            for coef_name in ["demi_cycles", "cycles", "translation",
                              "cosine"]:
                self.log(
                    f"loss_coef/{coef_name}",
                    getattr(self, f"loss_coef_{coef_name}"),
                    add_dataloader_idx=False,
                    batch_size=batch_size
                )

        return losses["total"], losses

    def training_step(self, domains, batch_idx):
        total_loss, _ = self.step(
            self.domains.encode(domains),
            mode="train"
        )
        return total_loss

    def validation_step(self, domains, batch_idx, dataset_idx=0):
        prefix = "in_dist/" if dataset_idx == 0 else "ood/"
        total_loss, _ = self.step(
            self.domains.encode(domains),
            mode="val",
            prefix=prefix
        )
        return total_loss

    def test_step(self, domains, batch_idx, dataset_idx=0):
        prefix = "in_dist/" if dataset_idx == 0 else "ood/"
        total_loss, _ = self.step(
            self.domains.encode(domains),
            mode="test", prefix=prefix
        )
        return total_loss

    def log_domains(
        self, logger, examples: DomainItems, slug="val",
        max_examples=None
    ):
        latents = self.domains.encode(examples)
        latents_sub_parts = {
            key: latent.sub_parts for key, latent in
            latents.items()
        }

        for domain_name, latent in latents.items():
            # Demi cycles
            predictions = self.domains.adapt(
                self.predict(self.project(latents_sub_parts, [domain_name]))
            )
            self.domains[domain_name].log_domain_from_latent(
                logger, predictions[domain_name],
                f"{slug}/demi_cycles/{domain_name}", max_examples
            )

            for domain_name_2 in latents.keys():
                if domain_name_2 != domain_name:
                    # Full cycles
                    cycle_predictions = self.predict(
                        self.project(predictions, [domain_name_2])
                    )
                    self.domains[domain_name].log_domain_from_latent(
                        logger, cycle_predictions[domain_name],
                        f"{slug}/cycles/{domain_name}_through_{domain_name_2}",
                        max_examples
                    )

                    # Translations
                    self.domains[domain_name_2].log_domain_from_latent(
                        logger, predictions[domain_name_2],
                        f"{slug}/translation/{domain_name}_to_{domain_name_2}",
                        max_examples,
                    )

    def log_original_domains(
        self, logger, examples: DomainItems, slug="val", max_examples=None
    ):
        if self.current_epoch == 0:
            with log_if_save_last_images(logger):
                with log_if_save_last_tables(logger):
                    for domain_name, domain_example in examples.items():
                        self.domains[domain_name].log_domain(
                            logger, domain_example.sub_parts,
                            f"{slug}/original/domain_{domain_name}",
                            max_examples
                        )

    def on_train_start(self) -> None:
        if self.domain_examples is None:
            return
        for mode, domain_examples in self.domain_examples.items():
            for logger in self.loggers:
                with self.domains.pass_through(False):
                    if (
                            hasattr(self.trainer.datamodule, "ood_boundaries")
                            and self.trainer.datamodule.ood_boundaries is
                            not None
                    ):
                        logger.log_hyperparams(
                            {
                                "ood_boundaries":
                                    self.trainer.datamodule.ood_boundaries
                            }
                        )
                    for dist, dist_examples in domain_examples.items():
                        self.log_original_domains(
                            logger, dist_examples,
                            f"{mode}/{dist}"
                        )

    def epoch_end(self, mode="val", log_train=True):
        domain_examples = self.domain_examples[mode]
        for logger in self.loggers:
            with self.domains.pass_through(False):
                for dist, dist_examples in domain_examples.items():
                    self.log_domains(
                        logger, dist_examples, f"{mode}/{dist}"
                    )

            if log_train:
                self.epoch_end("train", log_train=False)

    def validation_epoch_end(self, outputs):
        self.epoch_end("val", log_train=True)

    def test_epoch_end(self, outputs):
        self.epoch_end("test", log_train=False)

    def on_train_epoch_start(self):
        self.domains.eval()
        self.schedule_loss_coef_step()

    def configure_optimizers(self):
        params = []
        for model_type in ["encoders", "decoders"]:
            for domain_name, encoder in getattr(self, model_type).items():
                params.append(
                    {
                        "params": encoder.parameters(),
                        "lr": self.hparams.optim_lr[model_type],
                        "weight_decay": self.hparams.optim_weight_decay
                    }
                )
        optimizer = torch.optim.Adam(params)

        scheduler_interval = self.hparams.scheduler_interval
        scheduler_step = self.hparams.scheduler_step
        scheduler_gamma = self.hparams.scheduler_gamma
        if self.hparams.scheduler_mode == SchedulerMode.adaptive:
            # Convert into step interval if adaptive mode.
            size_dataset = len(self.trainer.datamodule.train_set["sync_"])
            batch_size = self.trainer.datamodule.batch_size
            if scheduler_interval == SchedulerInterval.step:
                n_step_per_epoch = int(size_dataset / batch_size)
                scheduler_step /= n_step_per_epoch
            # If less data, we need to do more scheduler steps. Must depend
            # on the synchronised data
            prop_labelled_images = (
                    1. - self.trainer.datamodule.prop_sync_domains["all"]
            )
            steps_per_new_epoch = int(
                scheduler_step * (
                        size_dataset * prop_labelled_images) / batch_size
            )
            scheduler_step = max(1, steps_per_new_epoch)
            scheduler_interval = SchedulerInterval.step
            print(f"Scheduler will be updated every {scheduler_step} step(s).")

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, scheduler_step, scheduler_gamma
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": scheduler_interval.value,
                "frequency": 1
            }
        }

    def vis_to_text_accuracy(
        self, domain_start, domain_end, acc_fn, domain_start_data, targets
    ):
        # translate the visual domain to text domain
        predicted_t = self.translate(
            domain_start_data, domain_start, domain_end
        )
        prediction = self.domains[domain_end].decode(predicted_t)
        return self.domains[domain_end].compute_acc(
            acc_fn, prediction, targets
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if not hasattr(self.trainer.datamodule, "domain_examples"):
            return
        if stage in ["fit", "validate", "test"]:
            self.domain_examples = self.trainer.datamodule.domain_examples

    def _put_domain_examples_to_device(self) -> None:
        if self.domain_examples is None:
            return
        for dist_examples in self.domain_examples.values():
            for examples in dist_examples.values():
                for domain_examples in examples.values():
                    domain_examples.to_device(self.device)

    def on_fit_start(self) -> None:
        self._put_domain_examples_to_device()

    def on_validation_start(self) -> None:
        self._put_domain_examples_to_device()
