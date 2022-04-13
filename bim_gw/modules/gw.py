from typing import Optional

import numpy as np
import scipy
import torch
import torch.nn.functional as F
import torchmetrics
from matplotlib import pyplot as plt
from neptune.new.types import File
from pytorch_lightning import LightningModule
from torch import nn

from bim_gw.modules.utils import DomainDecoder, DomainEncoder
from bim_gw.modules.workspace_module import PassThroughWM
from bim_gw.utils.grad_norms import GradNormLogger
from bim_gw.utils.utils import val_or_default


def check_domains_eq(domains_ori, domains_comp):
    for o, r in zip(domains_ori.values(), domains_comp.values()):
        assert torch.eq(o[0], r[0]).all()
        if o[1] is not None:
            assert torch.eq(o[1], r[1]).all()


class GlobalWorkspace(LightningModule):
    def __init__(
            self, domain_mods, z_size, hidden_size,
            n_classes=1000,
            loss_coef_demi_cycles=1., loss_coef_cycles=1., loss_coef_supervision=1., loss_coef_cosine=0.,
            optim_lr=3e-4, optim_weight_decay=1e-5, scheduler_mode="fixed", scheduler_interval="epoch",
            scheduler_step=20, scheduler_gamma=0.5,
            domain_examples: Optional[dict] = None,
            monitor_grad_norms: bool = False
    ):

        super(GlobalWorkspace, self).__init__()
        self.save_hyperparameters(ignore=["domain_mods", "domain_examples"])
        self.automatic_optimization = False

        self.z_size = z_size
        self.hidden_size = hidden_size
        self.monitor_grad_norms = monitor_grad_norms

        for mod in domain_mods.values():
            mod.freeze()  # insures that all modules are frozen

        self.domain_mods = nn.ModuleDict(domain_mods)
        self.domain_names = list(domain_mods.keys())
        self.validation_example_list = None

        # Define encoders for translation
        self.encoders = nn.ModuleDict({item: DomainEncoder(mod.output_dims, self.hidden_size, self.z_size)
                                       for item, mod in domain_mods.items()})
        self.decoders = nn.ModuleDict({item: (DomainDecoder(self.z_size, self.hidden_size,
                                                            mod.output_dims, mod.decoder_activation_fn))
                                       for item, mod in domain_mods.items()})

        # Define losses
        self.loss_fn = {}
        for domain_name, domain in self.domain_mods.items():
            for k in range(len(domain.losses)):
                self.loss_fn[f"{domain_name}_{k}"] = domain.losses[k]

        # Accuracies
        train_accuracy_metrics = []
        val_in_dist_accuracy_metrics = []
        val_ood_accuracy_metrics = []
        self.accuracy_metrics_order = []
        for domain_name, mod in self.domain_mods.items():
            if mod.requires_acc_computation:
                for domain_name_start, mod_start in self.domain_mods.items():
                    if domain_name_start != domain_name:
                        train_accuracy_metrics.append(torchmetrics.Accuracy())
                        val_in_dist_accuracy_metrics.append(torchmetrics.Accuracy())
                        val_ood_accuracy_metrics.append(torchmetrics.Accuracy())
                        self.accuracy_metrics_order.append((domain_name_start, domain_name))

        self.train_accuracy_metrics = nn.ModuleList(train_accuracy_metrics)
        self.val_in_dist_accuracy_metrics = nn.ModuleList(val_in_dist_accuracy_metrics)
        self.val_ood_accuracy_metrics = nn.ModuleList(val_ood_accuracy_metrics)

        self.grad_norms_bin = GradNormLogger()

        # val sampling
        self.domain_examples = domain_examples
        if domain_examples is not None:
            self.validation_example_list = dict()
            for dist, example_dist_vecs in domain_examples.items():
                if example_dist_vecs is not None:
                    for key, example_vecs in example_dist_vecs[0].items():
                        assert key in self.domain_names, f"{key} is not a valid domain for validation examples."
                        if example_vecs is not None:
                            self.validation_example_list[key] = len(example_vecs)
                            for k, example_vec in enumerate(example_vecs):
                                if type(example_vec) is list:
                                    setattr(self, f"validation_{dist}_examples_domain_{key}_{k}", example_vec)
                                else:
                                    self.register_buffer(f"validation_{dist}_examples_domain_{key}_{k}", example_vec)

        self.rotation_error_val = []
        print("Global Workspace instantiated.")

    def encode(self, x, domain_name):
        return self.encoders[domain_name](x)

    def decode(self, z, domain_name):
        return self.decoders[domain_name](z)

    def project(self, latents, masked_domains=None):
        state = [
            self.encode(latents[domain_name], domain_name)
            for domain_name in self.domain_names
        ]
        masks = [masked_domains]
        state = torch.stack(state, dim=1)

        if masked_domains is not None:
            masks = torch.cat(masks, dim=1)
            state[masks, :] = 0.
        return torch.sigmoid(state.sum(dim=1))

    def predict(self, state):
        return {
            domain_name: self.decode(state, domain_name)
            for domain_name in self.domain_names
        }

    def encode_uni_modal(self, domains):
        """
        Encodes unimodal inputs to their unimodal latent version
        """
        out = dict()
        for domain_name, x in domains.items():
            z = []
            for zi in self.domain_mods[domain_name].encode(x):
                if zi.ndim == 1:
                    z.append(zi.reshape(-1, 1))
                else:
                    z.append(zi)
            out[domain_name] = z
        return out

    def decode_uni_modal(self, domains):
        """
        Encodes unimodal inputs to their unimodal latent version
        """
        out = dict()
        for domain_name, x in domains.items():
            z = self.domain_mods[domain_name].decode(x)
            out[domain_name] = z
        return out

    def forward(self, domains):
        """
        Projects from latent version of unimodal vectors to the global workspace.
        """
        out = dict()
        for domain_name, x in domains.items():
            out[domain_name] = self.encode(x, domain_name)
        return out

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

    def demi_cycle_loss(self, domains, coef=1.):
        loss = torch.tensor(0.).to(self.device)
        losses = {}
        losses_no_coefs = {}
        total = len(domains)
        for name, domain in domains.items():
            out = self.demi_cycle(domain, name)
            l = torch.tensor(0.).to(self.device)
            for k in range(len(domain)):
                loss_fn = self.loss_fn[f"{name}_{k}"]
                l += loss_fn(out[k], domain[k]).mean() / total
            losses[f"loss_demi_cycle_{name}"] = coef * l
            losses_no_coefs[f"loss_demi_cycle_{name}"] = l
            loss += losses[f"loss_demi_cycle_{name}"]
        losses["demi_cycle_loss"] = loss
        losses_no_coefs["demi_cycle_loss"] = torch.mean(torch.stack(list(losses_no_coefs.values())))
        return losses["demi_cycle_loss"], losses, losses_no_coefs

    def cycle_loss(self, domains, coefficients=1.):
        loss = torch.tensor(0.).to(self.device)
        losses = {}
        losses_no_coefs = {}
        n = len(domains)
        total = n * (n - 1)
        for domain_name_start, domain in domains.items():
            for domain_name_inter in domains.keys():
                if domain_name_start != domain_name_inter:
                    token = f"{domain_name_start}_through_{domain_name_inter}"
                    coef = 1.
                    if isinstance(coefficients, (int, float)):
                        coef = coefficients
                    elif token in coefficients:
                        coef = coefficients[token]

                    out = self.cycle(domain, domain_name_start, domain_name_inter)

                    l = torch.tensor(0.).to(self.device)
                    for k in range(len(domain)):
                        loss_fn = self.loss_fn[f"{domain_name_start}_{k}"]
                        l += loss_fn(out[k], domain[k]).mean() / total
                    losses[f"loss_cycle_{token}"] = coef * l
                    losses_no_coefs[f"loss_cycle_{token}"] = l
                    loss += losses[f"loss_cycle_{token}"]
        losses["cycle_loss"] = loss
        losses_no_coefs["cycle_loss"] = torch.mean(torch.stack(list(losses_no_coefs.values())))
        return losses["cycle_loss"], losses, losses_no_coefs

    def supervision_loss(self, sync_domains, coefficients=1.):
        loss = torch.tensor(0.).to(self.device)
        losses = {}
        losses_no_coefs = {}
        total = 0
        for domain_name_1, domain_1 in sync_domains.items():
            for domain_name_2, domain_2 in sync_domains.items():
                if domain_name_1 != domain_name_2:
                    # project domains into one another
                    pred_domain_2 = self.translate(domain_1, domain_name_1, domain_name_2)

                    for k in range(len(domain_2)):
                        if domain_2[k] is not None:
                            token = f"{domain_name_1}_to_{domain_name_2}_{k}"
                            coef = 1.
                            if isinstance(coefficients, (int, float)):
                                coef = coefficients
                            elif token in coefficients:
                                coef = coefficients[token]

                            loss_fn = self.loss_fn[f"{domain_name_2}_{k}"]
                            l = loss_fn(pred_domain_2[k], domain_2[k]).mean()
                            losses[f"loss_supervision_{token}"] = coef * l
                            losses_no_coefs[f"loss_supervision_{token}"] = l
                            loss += losses[f"loss_supervision_{token}"]
                            total += 1
        if total > 0:
            for name in losses.keys():
                losses[name] = losses[name] / total
                losses_no_coefs[name] = losses_no_coefs[name] / total
            losses["supervision_loss"] = loss / total
            losses_no_coefs["supervision_loss"] = torch.mean(torch.stack(list(losses_no_coefs.values())))
        else:
            losses["supervision_loss"] = loss
        return losses["supervision_loss"], losses, losses_no_coefs

    def cosine_loss(self, sync_domains, coefficients=1.):
        loss = torch.tensor(0.).to(self.device)
        losses = {}
        losses_no_coefs = {}
        cosine_sims = {}
        total = 0
        for domain_name_1, domain_1 in sync_domains.items():
            for domain_name_2, domain_2 in sync_domains.items():
                if domain_name_1 != domain_name_2:
                    # project domains into one another
                    latent_domain_1 = self.encode(domain_1, domain_name_1)
                    latent_domain_2 = self.encode(domain_2, domain_name_2)
                    cosine_sims[f"cosine_sim_s_{domain_name_1}-s_{domain_name_2}"] = torch.cosine_similarity(
                        latent_domain_1, latent_domain_2).mean()

                    token = f"s_{domain_name_1}-s_{domain_name_2}"
                    coef = 1.
                    if isinstance(coefficients, (int, float)):
                        coef = coefficients
                    elif token in coefficients:
                        coef = coefficients[token]

                    l = F.cosine_embedding_loss(latent_domain_1, latent_domain_2,
                                                torch.ones(latent_domain_1.size(0)).to(latent_domain_1.device))
                    losses[f"loss_cosine_{token}"] = coef * l
                    losses_no_coefs[f"loss_cosine_{token}"] = l
                    loss += losses[f"loss_cosine_{token}"]
                    total += 1
        if total > 0:
            for name in losses.keys():
                losses[name] = losses[name] / total
                losses_no_coefs[name] = losses_no_coefs[name] / total
            losses["cosine_loss"] = loss / total
            losses_no_coefs["cosine_loss"] = torch.mean(torch.stack(list(losses_no_coefs.values())))
        else:
            losses["cosine_loss"] = loss
        losses_no_coefs.update(cosine_sims)
        return losses["cosine_loss"], losses, losses_no_coefs

    def step(self, latents, latent_targets, mode="val", prefix=""):
        losses = dict()
        loss_no_coef = dict()

        demi_cycle_loss, l, l_no_coefs = self.demi_cycle_loss(latents, self.hparams.loss_coef_demi_cycles)
        losses.update(l)
        loss_no_coef.update(l_no_coefs)

        cycle_loss, l, l_no_coefs = self.cycle_loss(latents, self.hparams.loss_coef_cycles)
        losses.update(l)
        loss_no_coef.update(l_no_coefs)

        supervision_loss, l, l_no_coefs = self.supervision_loss(latent_targets, self.hparams.loss_coef_supervision)
        losses.update(l)
        loss_no_coef.update(l_no_coefs)

        cosine_loss, l, l_no_coefs = self.cosine_loss(latent_targets, self.hparams.loss_coef_cosine)
        losses.update(l)
        loss_no_coef.update(l_no_coefs)

        total_loss = demi_cycle_loss + supervision_loss + cycle_loss + cosine_loss
        total_loss_no_coef = loss_no_coef["demi_cycle_loss"] + loss_no_coef["cycle_loss"] + loss_no_coef[
            "supervision_loss"] + loss_no_coef["cosine_loss"]

        batch_size = latents[list(latents.keys())[0]][0].size(0)
        for name, loss in loss_no_coef.items():
            self.log(f"{mode}{prefix}_{name}", loss, logger=True,
                     add_dataloader_idx=False, batch_size=batch_size)
        self.log(f"{mode}{prefix}_total_loss", total_loss_no_coef, logger=True,
                 add_dataloader_idx=False, batch_size=batch_size)
        self.log(f"{mode}{prefix}_total_loss_with_coef", total_loss, logger=True,
                 add_dataloader_idx=False, batch_size=batch_size)

        # compute accuracies
        for acc_fn, (domain_name_start, domain_name) in zip(getattr(self, f"{mode}{prefix}_accuracy_metrics"),
                                                            self.accuracy_metrics_order):
            predicted_t = self.translate(latent_targets[domain_name_start], domain_name_start, domain_name)
            prediction = self.domain_mods[domain_name].decode(predicted_t)
            accuracy = self.domain_mods[domain_name].compute_acc(acc_fn, prediction,
                                                                 latent_targets[domain_name])
            self.log(f"{mode}{prefix}_acc_{domain_name_start}_to_{domain_name}", accuracy, logger=True,
                     on_step=(mode != "val"), on_epoch=(mode == "val"), add_dataloader_idx=False)
        return total_loss, losses

    def training_step(self, batch, batch_idx):
        domains, targets = batch[0], batch[1]
        opt = self.optimizers()
        # remove the sync batch

        opt.zero_grad()

        latents = self.encode_uni_modal(domains)
        latent_targets = self.encode_uni_modal(targets)

        total_loss, losses = self.step(latents, latent_targets, mode="train")

        if self.monitor_grad_norms:
            grad_norms = self.manual_backward_with_grad_norm_monitoring(losses)
            self.grad_norms_bin.log(grad_norms)
            for name, grad_norm in grad_norms.items():
                self.log(f"grad_norm_{name.replace('@', '_')}", grad_norm, logger=True)
        else:
            self.manual_backward(total_loss)

        opt.step()

        return total_loss

    def validation_step(self, domains, batch_idx, dataset_idx=0):
        domains, targets = domains[0], domains[1]
        latents = self.encode_uni_modal(domains)
        prefix = "_in_dist" if dataset_idx == 0 else "_ood"
        total_loss, losses = self.step(latents, latents, mode="val", prefix=prefix)

        # latent_start = self.domain_mods["v"].encode(domains["v"])
        # latent_end = self.translate(latent_start, "v", "t")
        # domain_end_pred = self.domain_mods["t"].decode(latent_end)
        # rotations_pred = domain_end_pred[1][:, 3].detach().cpu().numpy()
        # rotations_ori = domains["t"][1][:, 3].detach().cpu().numpy()
        # diff = rotations_pred - rotations_ori
        # diff = np.where(diff > np.pi, diff - 2 * np.pi, diff)
        # self.rotation_error_val.extend((diff * 180 / np.pi).tolist())
        return total_loss

    def test_step(self, domains, batch_idx, dataset_idx=0):
        latents = self.encode_uni_modal(domains)
        total_loss, losses = self.step(latents, latents, domains, mode="test")
        return total_loss

    def get_validation_examples(self, dist):
        domain_examples = {}
        for domain_name, n_items in self.validation_example_list.items():
            domain_example = [
                getattr(self, f"validation_{dist}_examples_domain_{domain_name}_{k}") for k in range(n_items)
            ]
            if len(domain_example) == 1:
                domain_example = domain_example[0]
            domain_examples[domain_name] = domain_example
        return domain_examples

    def log_images(self, logger, examples, slug="val", max_examples=None):
        if self.current_epoch == 0:
            for domain_name, domain_example in examples.items():
                self.domain_mods[domain_name].log_domain(logger, domain_example,
                                                         f"{slug}_original_domain_{domain_name}", max_examples)
                if domain_name == "v":
                    latent = self.domain_mods[domain_name].encode(domain_example).detach().cpu().numpy()
                    fig, axes = plt.subplots(1, latent.shape[1])
                    for k in range(latent.shape[1]):
                        l = latent[:, k]
                        axes[k].hist(l, 50, density=True)
                        x = np.linspace(-0.8, 0.8, 100)
                        axes[k].plot(x, scipy.stats.norm.pdf(x, 0, 1))
                    logger.log_image("original_v_hist", fig)
                    plt.close(fig)

        if len(self.rotation_error_val):
            fig = plt.figure()
            plt.hist(self.rotation_error_val, 50, density=True)
            logger.log_image("rotation_error_val", fig)
            plt.close(fig)
            self.rotation_error_val = []

        for domain_name, domain_example in examples.items():
            # Demi cycles
            latent_x = self.domain_mods[domain_name].encode(domain_example)
            latent_reconstructed = self.demi_cycle(latent_x, domain_name)
            x_reconstructed = self.domain_mods[domain_name].decode(latent_reconstructed)
            self.domain_mods[domain_name].log_domain(logger, x_reconstructed,
                                                     f"{slug}_demi_cycle_{domain_name}", max_examples)

            for domain_name_2, domain_example_2 in examples.items():
                if domain_name_2 != domain_name:
                    # Full cycles
                    latent_x = self.domain_mods[domain_name].encode(domain_example)
                    latent_reconstructed = self.cycle(latent_x, domain_name, domain_name_2)
                    x_reconstructed = self.domain_mods[domain_name].decode(latent_reconstructed)
                    self.domain_mods[domain_name].log_domain(logger, x_reconstructed,
                                                             f"{slug}_cycle_{domain_name}_through_{domain_name_2}",
                                                             max_examples)

                    # Translations
                    latent_start = self.domain_mods[domain_name].encode(domain_example)
                    latent_end = self.translate(latent_start, domain_name, domain_name_2)
                    domain_end_pred = self.domain_mods[domain_name_2].decode(latent_end)
                    self.domain_mods[domain_name_2].log_domain(
                        logger, domain_end_pred,
                        f"{slug}_translation_{domain_name}_to_{domain_name_2}",
                        max_examples
                    )
                    if domain_name == "t" and domain_name_2 == "v":
                        fig, axes = plt.subplots(1, latent_end.size(1))
                        for k in range(latent_end.size(1)):
                            l = latent_end.detach().cpu().numpy()[:, k]
                            x = np.linspace(-0.8, 0.8, 100)
                            axes[k].hist(l, 50, density=True)
                            axes[k].plot(x, scipy.stats.norm.pdf(x, 0, 1))
                        logger.log_image("decoded_v_hist", fig)
                        plt.close(fig)

    def validation_epoch_end(self, outputs):
        if self.domain_examples is not None:
            for logger in self.loggers:
                self.set_unimodal_pass_through(False)
                if self.current_epoch == 0:
                    if self.trainer.datamodule.ood_boundaries is not None:
                        self.logger.experiment["ood_boundaries"] = str(self.trainer.datamodule.ood_boundaries)
            # self.logger.experiment["grad_norm_array"].upload(File.as_html(self.grad_norms_bin.values(15)))
                for dist in ["in_dist", "ood"]:
                    if self.domain_examples[dist] is not None:
                        validation_examples = self.get_validation_examples(dist)

                        if self.validation_example_list is not None:
                        self.log_images(logger, validation_examples, f"val_{dist}")

                if self.domain_examples["train"] is not None:
                    train_examples = self.get_validation_examples("train")
                self.log_images(logger, train_examples, "train")

                self.set_unimodal_pass_through(True)

    def set_unimodal_pass_through(self, mode=True):
        for domain_mod in self.domain_mods.values():
            if isinstance(domain_mod, PassThroughWM):
                domain_mod.pass_through(mode)

    def on_train_epoch_start(self):
        self.domain_mods.eval()

    def configure_optimizers(self):
        params = []
        for model_type in ["encoders", "decoders"]:
            for domain_name, encoder in getattr(self, model_type).items():
                params.append({
                    "params": encoder.parameters(),
                    "lr": self.hparams.optim_lr[model_type],
                    "weight_decay": self.hparams.optim_weight_decay
                })
        optimizer = torch.optim.Adam(params)

        scheduler_interval = self.hparams.scheduler_interval
        scheduler_step = self.hparams.scheduler_step
        scheduler_gamma = self.hparams.scheduler_gamma
        if self.hparams.scheduler_mode == "adaptive":
            # Convert into step interval if adaptive mode.
            if scheduler_interval == "epoch":
                size_dataset = len(self.trainer.datamodule.shapes_train)
                batch_size = self.trainer.datamodule.batch_size
                n_step_per_epoch = int(size_dataset / batch_size)
                scheduler_interval = "step"
                scheduler_step *= n_step_per_epoch
            # If less data, we need to do more scheduler steps. Must depend on the synchronised data
            prop_labelled_image = 1. - self.trainer.datamodule.prop_sync_domains["all"]
            scheduler_step = int(scheduler_step / prop_labelled_image)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step, scheduler_gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler_config": {
                "scheduler": scheduler,
                "interval": scheduler_interval,
                "frequency": 1
            }
        }

    def vis_to_text_accuracy(self, domain_start, domain_end, acc_fn, domain_start_data, targets):
        # translate the visual domain to text domain
        predicted_t = self.translate(domain_start_data, domain_start, domain_end)
        prediction = self.domain_mods[domain_end].decode(predicted_t)
        return self.domain_mods[domain_end].compute_acc(acc_fn, prediction, targets)

    def manual_backward_with_grad_norm_monitoring(self, losses):
        """
        Args:
            losses: Different losses to monitor separately.

        Returns: Gradient norms for each loss / sub-model couple.
        """
        grad_norms = {}
        last_grads = {}  # we need them to infer grad norm of each loss (and not accumulated gradients)
        for name, loss in losses.items():
            if name not in ["supervision_loss", "cycle_loss", "demi_cycle_loss"]:
                self.manual_backward(loss, retain_graph=True)
                for model_name in ["encoders", "decoders"]:
                    model = getattr(self, model_name)
                    for modality in model.keys():
                        param_group = f"{model_name}_{modality}"

                        grad_norms[f"{name}@{param_group}"] = torch.tensor(0.).type_as(loss) + sum([
                            # remove the already saved gradient that have already been counted in.
                            (p.grad.detach() - val_or_default(last_grads, f"{param_group}@{param_name}", 0)).norm()
                            for param_name, p in model[modality].named_parameters()
                            if p.grad is not None
                        ])
                        # assert grad_norms[f"{name}@{param_group}"] >= 0
                        if torch.isnan(grad_norms[f"{name}@{param_group}"]):
                            print(grad_norms)
                            print(f"{name}@{param_group}")
                            # print(grad_norms[f"{name}@{param_group}"])
                            # print(last_grads)

                        # Keep track of the value of the gradients to avoid counting
                        # them multiple times because of accumulation.
                        for param_name, p in model[modality].named_parameters():
                            if param_name not in last_grads:
                                last_grads[f"{param_group}@{param_name}"] = 0
                            if p.grad is not None:
                                last_grads[f"{param_group}@{param_name}"] += p.grad.detach()
        return grad_norms
