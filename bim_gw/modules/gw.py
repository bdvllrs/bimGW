from typing import Optional

import numpy as np
import scipy
import torch
import torchmetrics
from matplotlib import pyplot as plt
from neptune.new.types import File
from pytorch_lightning import LightningModule
from torch import nn

from bim_gw.modules.workspace_module import PassThroughWM
from bim_gw.utils.grad_norms import GradNormLogger


class DomainDecoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_size, out_dims=0, activation_fn=None):
        super(DomainDecoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size

        if isinstance(out_dims, int):
            out_dims = [out_dims]
        if not isinstance(activation_fn, (list, tuple)):
            activation_fn = [activation_fn]

        assert len(out_dims) == len(
            activation_fn), "The model is missing some loss_functions or output_dimensions for the outputs."

        self.out_dims = out_dims
        self.activation_fn = activation_fn

        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_size),
            # nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
        )

        self.encoder_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, pose_dim),
            )
            for pose_dim in self.out_dims
        ])

    def forward(self, x):
        out = self.encoder(x)
        outputs = []
        for block, activation_fn in zip(self.encoder_head, self.activation_fn):
            z = block(out)
            if activation_fn is not None:
                z = activation_fn(z)
            outputs.append(z)
        return outputs


class DomainEncoder(nn.Module):
    def __init__(self, in_dims, hidden_size, out_dim):
        super(DomainEncoder, self).__init__()
        if isinstance(in_dims, int):
            in_dims = [in_dims]
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(sum(self.in_dims), self.hidden_size),
            # nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.out_dim),
        )

    def forward(self, x):
        if len(self.in_dims) > 1:
            assert len(x) == len(self.in_dims), "Not enough values as input."
        x = torch.cat(x, dim=-1)
        out = self.encoder(x)
        return torch.tanh(out)


def check_domains_eq(domains_ori, domains_comp):
    for o, r in zip(domains_ori.values(), domains_comp.values()):
        assert torch.eq(o[0], r[0]).all()
        if o[1] is not None:
            assert torch.eq(o[1], r[1]).all()


class GlobalWorkspace(LightningModule):
    def __init__(
            self, domain_mods, z_size, hidden_size,
            n_classes=1000,
            loss_coef_demi_cycles=1, loss_coef_cycles=1, loss_coef_supervision=1,
            optim_lr=3e-4, optim_weight_decay=1e-5, scheduler_step=20, scheduler_gamma=0.5,
            domain_examples: Optional[dict] = None,
            monitor_grad_norms: bool = False
    ):

        super(GlobalWorkspace, self).__init__()
        self.save_hyperparameters()

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
                    for key in example_dist_vecs[0].keys():
                        assert key in self.domain_names, f"{key} is not a valid domain for validation examples."
                        if example_dist_vecs[0][key] is not None:
                            self.validation_example_list[key] = len(example_dist_vecs[0][key])
                            for k in range(len(example_dist_vecs[0][key])):
                                if type(example_dist_vecs[0][key][k]) is list:
                                    setattr(self, f"validation_{dist}_examples_domain_{key}_{k}", [
                                        example_dist_vecs[t][key][k] for t in range(len(example_dist_vecs))
                                    ])
                                else:
                                    self.register_buffer(
                                        f"validation_{dist}_examples_domain_{key}_{k}",
                                        torch.stack([
                                            example_dist_vecs[t][key][k] for t in range(len(example_dist_vecs))
                                        ], dim=0)
                                    )

        self.rotation_error_val = []
        print("Global Workspace instantiated.")

    def encode(self, x, domain_name):
        return self.encoders[domain_name](x)

    def decode(self, z, domain_name):
        return self.decoders[domain_name](z)

    def encode_modalities(self, domains):
        """
        Encodes unimodal inputs to their unimodal latent version
        """
        out = dict()
        for domain_name, x in domains.items():
            if domain_name != "_available_domains":
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

    def project(self, latents, masked_domains):
        state = torch.stack([
            self.encode(latents[domain_name], domain_name)
            for domain_name in self.domain_names
        ], dim=1)
        if masked_domains is not None:
            state[masked_domains, :] = 0.
        return state

    def predict(self, gw_state):
        return {
            domain_name: self.decode(gw_state, domain_name)
            for domain_name in self.domain_names
        }

    def forward(self, domains):
        """
        Projects from latent version of unimodal vectors to the global workspace.
        """
        out = dict()
        for domain_name, x in domains.items():
            out[domain_name] = self.encode(x, domain_name)
        return out

    def get_cycles(self, latents, available_domains, steps=2, masks=None):
        assert masks is None or len(masks) == steps
        assert steps >= 1
        unavailable_domains = torch.logical_not(available_domains)
        projection_input = latents
        projection_masks = masks
        if projection_masks is None:
            all_false = torch.full_like(unavailable_domains, False).to(self.device)
            # Default behavior: auto-regressive, always reuse prediction for the next projection
            projection_masks = [unavailable_domains] + [all_false for _ in range(steps - 1)]
        last_state = torch.zeros(unavailable_domains.size(0), len(latents), self.z_size).to(self.device)
        predictions = []
        for k in range(steps):
            mask = projection_masks[k]
            mask_inv = torch.logical_not(mask)
            # Project uni modal latent vectors into the GW
            last_state[mask_inv, :] = 0.
            # Sum last state and new projection according to the mask.
            last_state += self.project(projection_input, masked_domains=mask)
            # Predict all modalities from the GW state
            predictions.append(self.predict(last_state.sum(dim=1)))

            projection_input = predictions[-1]
        return predictions

    def cycle_loss(self, latents_ori, latents_pred, available_domains, coef=1., loss_name="cycle"):
        loss = torch.tensor(0.).to(self.device)
        losses = {}
        losses_no_coefs = {}
        total = len(latents_ori)
        for i, name in enumerate(sorted(latents_ori.keys())):
            latent_ori = latents_ori[name]
            latent_pred = latents_pred[name]

            l = torch.tensor(0.).to(self.device)
            for k in range(len(latent_ori)):
                # Only compute loss if there is at least one element containing modality "i".
                if available_domains[:, i].any():
                    loss_fn = self.loss_fn[f"{name}_{k}"]
                    pred = latent_pred[k][available_domains[:, i]]
                    ori = latent_ori[k][available_domains[:, i]]
                    l += loss_fn(pred, ori).mean() / total
            losses[f"loss_{loss_name}_{name}"] = coef * l
            losses_no_coefs[f"loss_{loss_name}_{name}"] = l
            loss += losses[f"loss_{loss_name}_{name}"]
        losses[f"{loss_name}_loss"] = loss
        losses_no_coefs[f"{loss_name}_loss"] = torch.mean(torch.stack(list(losses_no_coefs.values())))
        return losses[f"{loss_name}_loss"], losses, losses_no_coefs

    def step(self, latents, domains, mode="val", prefix=""):
        # One hot telling which modalities are in the input.
        # Unavailable modalities are artificially set to 0 in the "latents" dict.
        available_domains = domains["_available_domains"]
        unavailable_domains = torch.logical_not(available_domains)
        demi_cycle_predictions, cycle_predictions = self.get_cycles(latents, available_domains, steps=2,
                                                                    masks=[unavailable_domains, available_domains])
        # Compute all losses
        losses = dict()
        loss_no_coef = dict()

        demi_cycle_loss, l, l_no_coefs = self.cycle_loss(latents, demi_cycle_predictions, available_domains,
                                                         self.hparams.loss_coef_demi_cycles, "demi_cycle")
        losses.update(l)
        loss_no_coef.update(l_no_coefs)

        cycle_loss, l, l_no_coefs = self.cycle_loss(latents, cycle_predictions, available_domains,
                                                    self.hparams.loss_coef_cycles, "cycle")
        losses.update(l)
        loss_no_coef.update(l_no_coefs)

        total_loss = demi_cycle_loss + cycle_loss
        total_loss_no_coef = loss_no_coef["demi_cycle_loss"] + loss_no_coef["cycle_loss"]

        batch_size = latents[list(latents.keys())[0]][0].size(0)
        for name, loss in loss_no_coef.items():
            self.log(f"{mode}{prefix}_{name}", loss, logger=True,
                     add_dataloader_idx=False, batch_size=batch_size)
        self.log(f"{mode}{prefix}_total_loss", total_loss_no_coef, logger=True,
                 add_dataloader_idx=False, batch_size=batch_size)

        return total_loss, losses

    def training_step(self, domains, batch_idx):
        if batch_idx == 0 and self.current_epoch == 0:
            self.train_domain_examples = domains

        latents = self.encode_modalities(domains)
        total_loss, losses = self.step(latents, domains, mode="train")
        return total_loss

    def validation_step(self, domains, batch_idx, dataset_idx=0):
        latents = self.encode_modalities(domains)
        prefix = "_in_dist" if dataset_idx == 0 else "_ood"
        total_loss, losses = self.step(latents, domains, mode="val", prefix=prefix)
        return total_loss

    def test_step(self, domains, batch_idx, dataset_idx=0):
        latents = self.encode_modalities(domains)
        prefix = "_in_dist" if dataset_idx == 0 else "_ood"
        total_loss, losses = self.step(latents, domains, mode="test", prefix=prefix)
        return total_loss

    def get_validation_examples(self, dist):
        domain_examples = {}
        for domain_name, n_items in self.validation_example_list.items():
            domain_example = [
                getattr(self, f"validation_{dist}_examples_domain_{domain_name}_{k}") for k in range(n_items)
            ]
            domain_examples[domain_name] = domain_example
        return domain_examples

    def log_images(self, examples, slug="val", max_examples=None):
        if self.logger is not None:
            if self.current_epoch == 0:
                for domain_name, domain_example in examples.items():
                    self.domain_mods[domain_name].log_domain(self.logger, domain_example,
                                                             f"{slug}_original_domain_{domain_name}", max_examples)
                    if domain_name == "v":
                        latent = self.domain_mods[domain_name].encode(domain_example)[1].detach().cpu().numpy()
                        fig, axes = plt.subplots(1, latent.shape[1])
                        for k in range(latent.shape[1]):
                            l = latent[:, k]
                            axes[k].hist(l, 50, density=True)
                            x = np.linspace(-0.8, 0.8, 100)
                            axes[k].plot(x, scipy.stats.norm.pdf(x, 0, 1))
                        self.logger.experiment["original_v_hist"].log(File.as_image(fig))
                        plt.close(fig)

            if len(self.rotation_error_val):
                fig = plt.figure()
                plt.hist(self.rotation_error_val, 50, density=True)
                self.logger.experiment["rotation_error_val"].log(File.as_image(fig))
                plt.close(fig)
                self.rotation_error_val = []

            latents = self.encode_modalities(examples)
            available_domains = torch.stack([examples[domain_name][0] for domain_name in self.domain_names], dim=1).to(
                self.device, torch.bool)
            demi_cycle_predictions, cycle_predictions = self.get_cycles(latents, available_domains)

            for domain_name in examples.keys():
                x_reconstructed = self.domain_mods[domain_name].decode(demi_cycle_predictions[domain_name])
                self.domain_mods[domain_name].log_domain(self.logger, x_reconstructed,
                                                         f"{slug}_demi_cycle_{domain_name}", max_examples)

                x_reconstructed = self.domain_mods[domain_name].decode(cycle_predictions[domain_name])
                self.domain_mods[domain_name].log_domain(self.logger, x_reconstructed,
                                                         f"{slug}_cycle_{domain_name}",
                                                         max_examples)

    def validation_epoch_end(self, outputs):
        if self.logger is not None:
            self.set_unimodal_pass_through(False)
            if self.current_epoch == 0:
                if self.trainer.datamodule.ood_boundaries is not None:
                    self.logger.experiment["ood_boundaries"] = str(self.trainer.datamodule.ood_boundaries)
            self.logger.experiment["grad_norm_array"].upload(File.as_html(self.grad_norms_bin.values(15)))
            for dist in ["in_dist", "ood"]:
                if self.domain_examples[dist] is not None:
                    validation_examples = self.get_validation_examples(dist)

                    if self.validation_example_list is not None:
                        self.log_images(validation_examples, f"val_{dist}")

            if self.domain_examples["train"] is not None:
                train_examples = self.get_validation_examples("train")
                self.log_images(train_examples, "train")

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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.scheduler_step,
                                                    self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]

    def vis_to_text_accuracy(self, domain_start, domain_end, acc_fn, domain_start_data, targets):
        # translate the visual domain to text domain
        predicted_t = self.translate(domain_start_data, domain_start, domain_end)
        prediction = self.domain_mods[domain_end].decode(predicted_t)
        return self.domain_mods[domain_end].compute_acc(acc_fn, prediction, targets)
