from typing import Optional

import numpy as np
import scipy
import torch
import torchmetrics
from matplotlib import pyplot as plt
from neptune.new.types import File
from pytorch_lightning import LightningModule
from torch import nn

from bim_gw.utils.grad_norms import GradNormLogger
from bim_gw.utils.utils import val_or_default


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
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


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
        if isinstance(x, tuple) and len(x) == 1:
            x = x[0]
        elif isinstance(x, tuple):
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
            validation_domain_examples: Optional[dict] = None,
            monitor_grad_norms: bool = False
    ):

        super(GlobalWorkspace, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.z_size = z_size
        self.hidden_size = hidden_size
        self.monitor_grad_norms = monitor_grad_norms

        for mod in domain_mods.values():
            assert hasattr(mod, "z_size"), "Module must have a parameter z_size."
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
        self.validation_domain_examples = validation_domain_examples
        self.train_domain_examples = None
        if validation_domain_examples is not None:
            self.validation_example_list = dict()
            for dist, example_dist_vecs in validation_domain_examples.items():
                if example_dist_vecs is not None:
                    for key, example_vecs in example_dist_vecs.items():
                        assert key in self.domain_names, f"{key} is not a valid domain for validation examples."
                        if example_vecs is not None:
                            if not isinstance(example_vecs, tuple):
                                example_vecs = (example_vecs,)

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

    def encode_modalities(self, domains):
        """
        Encodes unimodal inputs to their unimodal latent version
        """
        out = dict()
        for domain_name, x in domains.items():
            if domain_name != "_available_domains":
                z = self.domain_mods[domain_name].encode(x)
                out[domain_name] = z
        return out

    def project(self, latents, masked_domains):
        state = torch.stack([
            self.encode(latents[domain_name], domain_name)
            for domain_name in self.domain_names
        ], dim=1)
        state[masked_domains, :] = 0.

        state = state.sum(dim=1)
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

    def get_cycles(self, latents, available_domains):
        unavailable_domains = torch.logical_not(available_domains)
        # Project uni modal latent vectors into the GW
        gw_state = self.project(latents, unavailable_domains)
        # Predict all modalities from the GW state
        demi_cycle_predictions = self.predict(gw_state)
        # Make full cycle by encoding the unavailable modalities into the GW and predicting again
        cycle_gw_state = gw_state + self.project(demi_cycle_predictions, available_domains)
        cycle_predictions = self.predict(cycle_gw_state)
        return demi_cycle_predictions, cycle_predictions

    def cycle_loss(self, latents_ori, latents_pred, available_domains, coef=1., loss_name="cycle"):
        loss = torch.tensor(0.).to(self.device)
        losses = {}
        losses_no_coefs = {}
        total = len(latents_ori)
        for i, name in enumerate(sorted(latents_ori.keys())):
            latent_ori = latents_ori[name]
            latent_pred = latents_pred[name]

            if not isinstance(latent_ori, tuple):
                latent_ori = (latent_ori,)
                latent_pred = (latent_pred,)

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
        demi_cycle_predictions, cycle_predictions = self.get_cycles(latents, available_domains)
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

        for name, loss in loss_no_coef.items():
            self.log(f"{mode}{prefix}_{name}", loss, logger=True, add_dataloader_idx=False)
        self.log(f"{mode}{prefix}_total_loss", total_loss_no_coef, logger=True, add_dataloader_idx=False)

        return total_loss, losses

    def training_step(self, domains, batch_idx):
        if batch_idx == 0 and self.current_epoch == 0:
            self.train_domain_examples = domains

        opt = self.optimizers()
        opt.zero_grad()

        latents = self.encode_modalities(domains)
        total_loss, losses = self.step(latents, domains, mode="train")

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
            if len(domain_example) == 1:
                domain_example = domain_example[0]
            domain_examples[domain_name] = domain_example
        return domain_examples

    def log_images(self, examples, slug="val", max_examples=None):
        if self.logger is not None:
            if self.current_epoch == 0:
                for domain_name, domain_example in examples.items():
                    self.domain_mods[domain_name].log_domain(self.logger, domain_example,
                                                             f"{slug}_original_domain_{domain_name}", max_examples)

                    if domain_name == "v":
                        latent = self.domain_mods[domain_name].encode(domain_example)
                        fig, axes = plt.subplots(1, latent.size(1))
                        for k in range(latent.size(1)):
                            l = latent.detach().cpu().numpy()[:, k]
                            x = np.linspace(-0.8, 0.8, 100)
                            axes[k].hist(l, 50, density=True)
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
            available_domains = torch.stack([examples[domain_name][0] for domain_name in self.domain_names], dim=1).to(self.device, torch.bool)
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
            if self.current_epoch == 0:
                if self.trainer.datamodule.ood_boundaries is not None:
                    self.logger.experiment["ood_boundaries"] = str(self.trainer.datamodule.ood_boundaries)
            self.logger.experiment["grad_norm_array"].upload(File.as_html(self.grad_norms_bin.values(15)))
            for dist in ["in_dist", "ood"]:
                if self.validation_domain_examples[dist] is not None:
                    validation_examples = self.get_validation_examples(dist)

                    if self.validation_example_list is not None:
                        self.log_images(validation_examples, f"val_{dist}")

            if self.train_domain_examples is not None:
                self.log_images(self.train_domain_examples, "train", 32)

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
