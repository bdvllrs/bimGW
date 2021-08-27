from typing import Optional

import torch
import torchmetrics
from neptune.new.types import File
from omegaconf import ListConfig
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F

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
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU()
        )

        self.encoder_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.BatchNorm1d(self.hidden_size),
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
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.out_dim)
        )

    def forward(self, x):
        if len(self.in_dims) > 1:
            assert len(x) == len(self.in_dims), "Not enough values as input."
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        elif isinstance(x, (list, tuple)):
            x = torch.cat(x, dim=-1)
        out = self.encoder(x)
        return torch.tanh(out)


def check_domains_eq(domains_ori, domains_comp):
    for o, r in zip(domains_ori.values(), domains_comp.values()):
        assert torch.eq(o[0], r[0]).all()
        if o[1] is not None:
            assert torch.eq(o[1], r[1]).all()


def cross_entropy(x, y):
    y = torch.argmax(y, 1)
    return F.cross_entropy(x, y)


def nll_loss(x, y):
    y = torch.argmax(y, 1)
    return F.nll_loss(x, y)


loss_functions = {
    "cosine": lambda x, y: -F.cosine_similarity(x, y),
    "mse": F.mse_loss,
    "cross_entropy": cross_entropy,
    "nll": nll_loss
}


class GlobalWorkspace(LightningModule):
    def __init__(
            self, domain_mods, z_size, hidden_size,
            n_classes=1000,
            loss_coef_demi_cycles=1, loss_coef_cycles=1, loss_coef_supervision=1,
            cycle_loss_fn="cosine", supervision_loss_fn=None,
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
        assert cycle_loss_fn in loss_functions, f"Cycle loss function {cycle_loss_fn} must be in {loss_functions.keys()}."
        self.cycle_loss_fn = loss_functions[cycle_loss_fn]

        self.supervision_loss_fn = {}
        for domain in self.domain_mods.keys():
            domain_supervision_loss_fn = "cosine" if supervision_loss_fn is None else supervision_loss_fn[domain]
            # Iterate over class and latent (they could have different loss fn)
            if not isinstance(domain_supervision_loss_fn, ListConfig):
                domain_supervision_loss_fn = (domain_supervision_loss_fn,)
            for k in range(len(domain_supervision_loss_fn)):
                assert domain_supervision_loss_fn[
                           k] in loss_functions, f"Supervision loss function {domain_supervision_loss_fn[k]} must be in {loss_functions.keys()}."
                self.supervision_loss_fn[f"{domain}_{k}"] = loss_functions[domain_supervision_loss_fn[k]]

        # Accuracies
        train_accuracy_metrics = []
        val_accuracy_metrics = []
        self.accuracy_metrics_order = []
        for domain_name, mod in self.domain_mods.items():
            if mod.requires_acc_computation:
                for domain_name_start, mod_start in self.domain_mods.items():
                    if domain_name_start != domain_name:
                        train_accuracy_metrics.append(torchmetrics.Accuracy())
                        val_accuracy_metrics.append(torchmetrics.Accuracy())
                        self.accuracy_metrics_order.append((domain_name_start, domain_name))

        self.train_accuracy_metrics = nn.ModuleList(train_accuracy_metrics)
        self.val_accuracy_metrics = nn.ModuleList(val_accuracy_metrics)

        self.grad_norms_bin = GradNormLogger()

        # val sampling
        if validation_domain_examples is not None:
            self.validation_example_list = dict()
            for key, example_vecs in validation_domain_examples.items():
                assert key in self.domain_names, f"{key} is not a valid domain for validation examples."
                if not isinstance(example_vecs, (tuple, list)):
                    example_vecs = [example_vecs]

                self.validation_example_list[key] = len(example_vecs)
                for k, example_vec in enumerate(example_vecs):
                    self.register_buffer(f"validation_examples_domain_{key}_{k}", example_vec)
        print("done!")

    def project(self, domains):
        """
        Projects unimodal domains to global workspace
        """
        out = dict()
        for domain_name, x in domains.items():
            z = self.domain_mods[domain_name].encode(x)
            out[domain_name] = z
        return out

    def forward(self, domains):
        pass

    def translate(self, x, domain_name_start, domain_name_target):
        """
        Translates x from domain1 to domain2
        """
        z = self.encoders[domain_name_start](x)
        return self.decoders[domain_name_target](z)

    def demi_cycle(self, x, domain_inter):
        return self.translate(x, domain_inter, domain_inter)

    def cycle(self, x, domain_name_start, domain_name_inter):
        z = self.translate(x, domain_name_start, domain_name_inter)
        return self.translate(z, domain_name_inter, domain_name_start)

    def demi_cycle_loss(self, domains, coefficients=1.):
        loss = torch.tensor(0.).to(self.device)
        losses = {}
        total = len(domains)
        for name, domain in domains.items():

            coef = 1.
            if isinstance(coefficients, (int, float)):
                coef = coefficients
            elif name in coefficients:
                coef = coefficients[name]

            out = self.demi_cycle(domain, name)

            if not isinstance(domain, (list, tuple)):
                assert not isinstance(domain, (list, tuple))
                domain = [domain]
                out = [out]

            l = torch.tensor(0.).to(self.device)
            for k in range(len(domain)):
                l += coef * self.cycle_loss_fn(out[k], domain[k]).mean() / total
            losses[f"loss_demi_cycle_{name}"] = l
            loss += l
        losses["demi_cycle_loss"] = loss
        return losses["demi_cycle_loss"], losses

    def cycle_loss(self, domains, coefficients=1.):
        loss = torch.tensor(0.).to(self.device)
        losses = {}
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

                    if not isinstance(domain, (list, tuple)):
                        assert not isinstance(domain, (list, tuple))
                        domain = [domain]
                        out = [out]

                    l = torch.tensor(0.).to(self.device)
                    for k in range(len(domain)):
                        l += coef * self.cycle_loss_fn(out[k], domain[k]).mean() / total
                    losses[f"loss_cycle_{token}"] = l
                    loss += l
        losses["cycle_loss"] = loss
        return losses["cycle_loss"], losses

    def supervision_loss(self, sync_domains, coefficients=1.):
        loss = torch.tensor(0.).to(self.device)
        losses = {}
        total = 0
        for domain_name_1, domain_1 in sync_domains.items():
            for domain_name_2, domain_2 in sync_domains.items():
                if domain_name_1 != domain_name_2:
                    # project domains into one another
                    pred_domain_2 = self.translate(domain_1, domain_name_1, domain_name_2)
                    if not isinstance(domain_2, (list, tuple)):
                        assert not isinstance(domain_2, (list, tuple))
                        domain_2 = [domain_2]
                        pred_domain_2 = [pred_domain_2]
                    for k in range(len(domain_2)):
                        if domain_2[k] is not None:
                            token = f"{domain_name_1}_to_{domain_name_2}_{k}"
                            coef = 1.
                            if isinstance(coefficients, (int, float)):
                                coef = coefficients
                            elif token in coefficients:
                                coef = coefficients[token]

                            loss_fn = self.supervision_loss_fn[f"{domain_name_2}_{k}"]
                            l = coef * loss_fn(pred_domain_2[k], domain_2[k]).mean()
                            losses[f"loss_supervision_{token}"] = l
                            loss += l
                            total += 1
        losses["supervision_loss"] = loss
        if total > 0:
            for name in losses.keys():
                losses[name] = losses[name] / total
            losses["supervision_loss"] = loss / total
        return losses["supervision_loss"], losses

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        # remove the sync batch
        domains = {key: val for key, val in batch.items() if key != "sync_"}
        sync_supervision = batch["sync_"]  # Sparse cross-modal supervision
        latents = self.project(domains)
        ori_latents = latents.copy()

        losses = dict()

        demi_cycle_loss, l = self.demi_cycle_loss(latents, self.hparams.loss_coef_demi_cycles)
        losses.update(l)
        check_domains_eq(ori_latents, latents)

        cycle_loss, l = self.cycle_loss(latents, self.hparams.loss_coef_cycles)
        losses.update(l)
        check_domains_eq(ori_latents, latents)

        sync_latents = self.project(sync_supervision)
        supervision_loss, l = self.supervision_loss(sync_latents, self.hparams.loss_coef_supervision)
        losses.update(l)

        total_loss = demi_cycle_loss + cycle_loss + supervision_loss

        for name, loss in losses.items():
            self.log(f"train_{name}", loss, logger=True)
        self.log(f"train_total_loss", total_loss, logger=True)

        # compute accuracies
        for train_acc_fn, (domain_name_start, domain_name) in zip(self.train_accuracy_metrics,
                                                                  self.accuracy_metrics_order):
            predicted_t = self.translate(sync_latents[domain_name_start], domain_name_start, domain_name)
            prediction = self.domain_mods[domain_name].decode(predicted_t)
            accuracy = self.domain_mods[domain_name].compute_acc(train_acc_fn, prediction,
                                                                 sync_supervision[domain_name])
            self.log(f"train_acc_{domain_name_start}_to_{domain_name}", accuracy, on_step=True, on_epoch=False)

        if self.monitor_grad_norms:
            grad_norms = self.manual_backward_with_grad_norm_monitoring(losses)
            self.grad_norms_bin.log(grad_norms)
            for name, grad_norm in grad_norms.items():
                self.log(f"grad_norm_{name.replace('@', '_')}", grad_norm, logger=True)
        else:
            self.manual_backward(total_loss)

        opt.step()
        return total_loss

    def validation_step(self, domains, batch_idx):
        latents = self.project(domains)
        ori_latents = latents.copy()
        losses = dict()
        demi_cycle_loss, l = self.demi_cycle_loss(latents, self.hparams.loss_coef_demi_cycles)
        losses.update(l)

        check_domains_eq(ori_latents, latents)
        cycle_loss, l = self.cycle_loss(latents, self.hparams.loss_coef_cycles)
        losses.update(l)
        check_domains_eq(ori_latents, latents)
        supervision_loss, l = self.supervision_loss(latents, self.hparams.loss_coef_supervision)
        losses.update(l)

        total_loss = demi_cycle_loss + cycle_loss + supervision_loss

        for name, loss in losses.items():
            self.log(f"val_{name}", loss, logger=True)
        self.log(f"val_total_loss", total_loss, logger=True)

        # compute accuracies
        for val_acc_fn, (domain_name_start, domain_name) in zip(self.val_accuracy_metrics,
                                                                self.accuracy_metrics_order):
            predicted_t = self.translate(latents[domain_name_start], domain_name_start, domain_name)
            prediction = self.domain_mods[domain_name].decode(predicted_t)
            accuracy = self.domain_mods[domain_name].compute_acc(val_acc_fn, prediction, domains[domain_name])
            self.log(f"val_acc_{domain_name_start}_to_{domain_name}", accuracy, on_step=True, on_epoch=True)

        return total_loss

    def get_validation_examples(self):
        domain_examples = {}
        for domain_name, n_items in self.validation_example_list.items():
            domain_example = [
                getattr(self, f"validation_examples_domain_{domain_name}_{k}") for k in range(n_items)
            ]
            if len(domain_example) == 1:
                domain_example = domain_example[0]
            domain_examples[domain_name] = domain_example
        return domain_examples

    def validation_epoch_end(self, outputs):
        if self.logger is not None:
            validation_examples = self.get_validation_examples()

            self.logger.experiment["grad_norm_array"].upload(File.as_html(self.grad_norms_bin.values(15)))

            if self.validation_example_list is not None:
                if self.current_epoch == 0:

                    for domain_name, domain_example in validation_examples.items():
                        self.domain_mods[domain_name].log_domain(self.logger, domain_example,
                                                                 f"val_original_domain_{domain_name}")

                for domain_name, domain_example in validation_examples.items():
                    # Demi cycles
                    latent_x = self.domain_mods[domain_name].encode(domain_example)
                    latent_reconstructed = self.demi_cycle(latent_x, domain_name)
                    x_reconstructed = self.domain_mods[domain_name].decode(latent_reconstructed)
                    self.domain_mods[domain_name].log_domain(self.logger, x_reconstructed,
                                                             f"val_demi_cycle_{domain_name}")

                    for domain_name_2, domain_example_2 in validation_examples.items():
                        if domain_name_2 != domain_name:
                            # Full cycles
                            latent_x = self.domain_mods[domain_name].encode(domain_example)
                            latent_reconstructed = self.cycle(latent_x, domain_name, domain_name_2)
                            x_reconstructed = self.domain_mods[domain_name].decode(latent_reconstructed)
                            self.domain_mods[domain_name].log_domain(self.logger, x_reconstructed,
                                                                     f"val_cycle_{domain_name}_through_{domain_name_2}")

                            # Translations
                            latent_start = self.domain_mods[domain_name].encode(domain_example)
                            latent_end = self.translate(latent_start, domain_name, domain_name_2)
                            domain_end_pred = self.domain_mods[domain_name_2].decode(latent_end)
                            self.domain_mods[domain_name_2].log_domain(
                                self.logger, domain_end_pred,
                                f"val_translation_{domain_name}_to_{domain_name_2}"
                            )

    def on_train_epoch_start(self):
        self.domain_mods.eval()

    def configure_optimizers(self):
        params = []
        for domain_name, encoder in self.encoders.items():
            params.append({
                'params': encoder.parameters(), "lr": self.hparams.optim_lr,
                "weight_decay": self.hparams.optim_weight_decay
            })
        for domain_name, decoder in self.decoders.items():
            params.append({
                'params': decoder.parameters(), "lr": self.hparams.optim_lr,
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
                        assert grad_norms[f"{name}@{param_group}"] >= 0

                        # Keep track of the value of the gradients to avoid counting
                        # them multiple times because of accumulation.
                        for param_name, p in model[modality].named_parameters():
                            if param_name not in last_grads:
                                last_grads[f"{param_group}@{param_name}"] = 0
                            if p.grad is not None:
                                last_grads[f"{param_group}@{param_name}"] += p.grad.detach()
        return grad_norms
