from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule
from torch import nn

from bim_gw.modules.utils import DomainDecoder, DomainEncoder, mask_predictions
from bim_gw.modules.workspace_module import PassThroughWM
from bim_gw.utils.grad_norms import GradNormLogger


def check_domains_eq(domains_ori, domains_comp):
    for o, r in zip(domains_ori.values(), domains_comp.values()):
        assert torch.eq(o[0], r[0]).all()
        if o[1] is not None:
            assert torch.eq(o[1], r[1]).all()


def split_domains_available_domains(domains):
    indicators = {domain_name: domain[0].to(torch.bool) for domain_name, domain in domains.items()}
    domains = {domain_name: domain[1:] for domain_name, domain in domains.items()}
    return indicators, domains


class GlobalWorkspace(LightningModule):
    def __init__(
            self, domain_mods, z_size, hidden_size, n_layers_encoder, n_layers_decoder, n_layers_decoder_head,
            n_classes=1000,
            loss_coef_demi_cycles=1., loss_coef_cycles=1., loss_coef_translation=1., loss_coef_cosine=0.,
            loss_coef_contrastive=0., optim_lr=3e-4, optim_weight_decay=1e-5, scheduler_mode="fixed",
            scheduler_interval="epoch", scheduler_step=20, scheduler_gamma=0.5, loss_schedules=None,
            domain_examples: Optional[dict] = None,
            monitor_grad_norms: bool = False,
            remove_sync_domains=None,
            nulll_items=None
    ):

        super(GlobalWorkspace, self).__init__()
        # self.automatic_optimization = False

        self.loss_coef_demi_cycles = loss_coef_demi_cycles
        self.loss_coef_cycles = loss_coef_cycles
        self.loss_coef_translation = loss_coef_translation
        self.loss_coef_cosine = loss_coef_cosine
        self.loss_schedules = loss_schedules if loss_schedules is not None else {}
        self.remove_sync_domains = remove_sync_domains if remove_sync_domains is not None else []
        self.remove_sync_domains_names = [f"{n1}-{n2}" for n1, n2 in self.remove_sync_domains] + [f"{n2}-{n1}" for
                                                                                                  n1, n2 in
                                                                                                  self.remove_sync_domains]

        self.z_size = z_size
        self.hidden_size = hidden_size
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_layers_decoder_head = n_layers_decoder_head
        self.monitor_grad_norms = monitor_grad_norms

        for mod in domain_mods.values():
            mod.freeze()  # insures that all modules are frozen

        self.domain_mods = nn.ModuleDict(domain_mods)
        self.domain_names = list(domain_mods.keys())
        self.validation_example_list = None

        self.hparams["contrastive_logit_scale"] = {
            f"{n1}-{n2}": torch.ones([]) * np.log(1 / 0.07)
            for i, n1 in enumerate(self.domain_names)
            for j, n2 in enumerate(self.domain_names)
            if i < j
        }

        # Define encoders for translation
        self.encoders = nn.ModuleDict({item: DomainEncoder(mod.output_dims, self.hidden_size,
                                                           self.z_size, self.n_layers_encoder)
                                       for item, mod in domain_mods.items()})
        self.decoders = nn.ModuleDict({item: (DomainDecoder(self.z_size, self.hidden_size,
                                                            self.n_layers_decoder, self.n_layers_decoder_head,
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

        self.null_items = nulll_items
        # val sampling
        self.domain_examples = domain_examples
        if domain_examples is not None:
            self.validation_example_list = dict()
            for set_name, example_dist_vecs in domain_examples.items():
                for dist in range(2):
                    dist_name = "in_dist" if dist == 0 else "ood"
                    if example_dist_vecs[dist] is not None:
                        for key, example_vecs in example_dist_vecs[dist].items():
                            assert key in self.domain_names, f"{key} is not a valid domain for validation examples."
                            if example_vecs is not None:
                                self.validation_example_list[key] = len(example_vecs)
                                for k, example_vec in enumerate(example_vecs):
                                    if type(example_vec) is list:
                                        setattr(self, f"{set_name}_{dist_name}_examples_domain_{key}_{k}", example_vec)
                                    else:
                                        self.register_buffer(f"{set_name}_{dist_name}_examples_domain_{key}_{k}",
                                                             example_vec)

        self.rotation_error_val = []

        self.save_hyperparameters(ignore=["domain_mods", "domain_examples", "loss_schedules"])
        print("Global Workspace instantiated.")

    def setup(self, stage=None):
        self.collate_fn = self.trainer.datamodule.train_dataloader().collate_fn

    def encode(self, x, domain_name, add_tanh=True):
        pre_act = self.encoders[domain_name](x)
        if add_tanh:
            return torch.tanh(pre_act)
        return pre_act

    def decode(self, z, domain_name):
        return self.decoders[domain_name](z)

    def get_null_latent(self, batch_size, domain_name, original_latent=False):
        if original_latent:
            items = list(self.null_items[domain_name])
        else:
            items = list(self.trainer.datamodule.shapes_train.data_fetchers[domain_name].get_null_item())
        batch = [items for k in range(batch_size)]
        x = self.collate_fn(batch)[1:]
        for k in range(len(x)):
            if isinstance(x[k], torch.Tensor):
                x[k] = x[k].to(self.device)
        return self.encode_uni_modal({domain_name: x})[domain_name]

    def project(self, latents):
        projected_domains = {}
        for domain, latent in latents.items():
            projected_domains[domain] = self.encode(latents[domain], domain, add_tanh=False)
        return projected_domains

    def combine(self, states, selected_domains=None):
        pre_act = None
        for domain in states.keys():
            if selected_domains is None or domain in selected_domains:
                if pre_act is None:
                    pre_act = states[domain]
                else:
                    pre_act += states[domain]
        assert pre_act is not None
        return torch.tanh(pre_act)

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

    def adapt(self, latents):
        return {domain: self.domain_mods[domain].adapt(latent) for domain, latent in latents.items()}

    def forward(self, domains):
        latents = self.encode_uni_modal(domains)
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
                if (self.current_epoch != 0) and (self.current_epoch % self.loss_schedules[loss_name]["step"] == 0):
                    setattr(self, f"loss_coef_{loss_name}", coef * self.loss_schedules[loss_name]["gamma"])

    def cosine_loss(self, states):
        losses = []
        indiv_losses = {}
        for domain_name in states.keys():
            domain_name_1, domain_name_2 = domain_name.split("-")
            latent_domain_1 = states[domain_name]
            latent_domain_2 = states[f"{domain_name_2}-{domain_name_1}"]
            # project domains into one another
            indiv_losses[f"cosine_similarity_s_{domain_name_1}-s_{domain_name_2}"] = torch.cosine_similarity(
                latent_domain_1, latent_domain_2).mean()

            l = F.cosine_embedding_loss(latent_domain_1, latent_domain_2,
                                        torch.ones(latent_domain_1.size(0)).to(latent_domain_1.device))
            indiv_losses[f"cosine_loss_s_{domain_name_1}-s_{domain_name_2}"] = l
            losses.append(l)
        if len(losses):
            indiv_losses["cosine"] = torch.stack(losses, dim=0).mean()
            return indiv_losses
        return {"cosine": torch.tensor(0.).to(self.device)}

    def contrastive_loss(self, states, masks):
        losses = []
        indiv_losses = {}
        for domain_name_1 in states.keys():
            for domain_name_2 in states.keys():
                if f"{domain_name_1}-{domain_name_2}" in self.hparams.contrastive_logit_scale.keys() and domain_name_2 != domain_name_1:
                    mask = torch.logical_and(masks[domain_name_1], masks[domain_name_2])
                    latent_domain_1 = states[domain_name_1][mask]
                    latent_domain_2 = states[domain_name_2][mask]
                    # project domains into one another
                    latent_features_d2 = latent_domain_2 / latent_domain_2.norm(dim=1, keepdim=True)
                    latent_features_d1 = latent_domain_1 / latent_domain_1.norm(dim=1, keepdim=True)
                    logit_scale = self.hparams.contrastive_logit_scale[f"{domain_name_1}-{domain_name_2}"].exp()
                    logits = logit_scale * latent_features_d1 @ latent_features_d2.t()
                    labels = torch.arange(latent_domain_1.size(0)).to(logits.device)
                    loss_d1 = F.cross_entropy(logits, labels)
                    loss_d2 = F.cross_entropy(logits.t(), labels)
                    loss = .5 * (loss_d1 + loss_d2)
                    indiv_losses[f"contrastive_s_{domain_name_1}-s_{domain_name_2}"] = loss
                    losses.append(loss)
        if len(losses):
            indiv_losses["contrastive"] = torch.stack(losses, dim=0).mean()
            return indiv_losses
        return {"contrastive": torch.tensor(0.).to(self.device)}

    def loss(self, predictions, targets, prefix=""):
        losses = []
        indiv_losses = {}
        for domain_name in predictions.keys():
            prediction, target = predictions[domain_name], targets[domain_name]
            loss = torch.tensor(0.).to(prediction[0].device)
            for k in range(len(prediction)):
                token = f"{prefix}/domain_{domain_name}_{k}"
                loss_domain = domain_name
                if "-" in domain_name:
                    loss_domain = domain_name.split("-")[0]
                loss_fn = self.loss_fn[f"{loss_domain}_{k}"]
                l = loss_fn(prediction[k], target[k]).mean()
                loss += l
                indiv_losses[token] = l
            token = f"{prefix}/domain_{domain_name}"
            indiv_losses[token] = loss
            losses.append(loss)
        if len(losses):
            indiv_losses[prefix] = torch.stack(losses, dim=0).mean()
            return indiv_losses
        return {prefix: torch.tensor(0.).to(self.device)}

    def step(self, available_domains, latents, mode="val", prefix=""):
        prop_sync = torch.min(available_domains['v'], available_domains['t']).sum() / available_domains['v'].size(0)
        self.log(f"{mode}/{prefix}prop_sync_batch", prop_sync, on_step=True, on_epoch=False)

        latent_predictions = {
            "tr": {},
            "cy": {},
            "dcy": {}
        }
        latent_targets = {}
        states = {}
        masks = {}
        null_latents = {domain_name: self.get_null_latent(available_domains["v"].size(0), domain_name)
                        for domain_name in available_domains.keys()}

        for domain_name_target, latent_target in latents.items():
            mask = available_domains[domain_name_target]
            masks[domain_name_target] = mask
            if mask.any():
                latent_targets[domain_name_target] = [latents[domain_name_target][k][mask]
                                                      for k in range(len(latents[domain_name_target]))]

                # Demi-cycles
                masked_latents = {
                    domain_name: ([latents[domain_name][k][mask] for k in range(len(latents[domain_name]))]
                                  if domain_name == domain_name_target
                                  else [null_latents[domain_name][k][mask]
                                        for k in range(len(latents[domain_name]))])
                    for domain_name in latents.keys()
                }
                state_domain = self.project(masked_latents)
                state = self.combine(state_domain)
                last_latents = self.predict(state)
                latent_predictions["dcy"][domain_name_target] = last_latents[domain_name_target]

                # Cycle
                masked_latents = {
                    domain_name: ([last_latents[domain_name][k] for k in range(len(last_latents[domain_name]))]
                                  if domain_name != domain_name_target
                                  else [null_latents[domain_name][k][mask]
                                        for k in range(len(last_latents[domain_name]))])
                    for domain_name in last_latents.keys()
                }
                state_domain = self.project(masked_latents)
                state = self.combine(state_domain)
                latent_predictions["cy"][domain_name_target] = self.predict(state)[domain_name_target]

                # Translation
                masked_latents = {
                    domain_name: ([latents[domain_name][k][mask] for k in range(len(latents[domain_name]))]
                                  if domain_name != domain_name_target
                                  else [null_latents[domain_name][k][mask]
                                        for k in range(len(latents[domain_name]))])
                    for domain_name in latents.keys()
                }
                state = self.combine(self.project(masked_latents))
                predictions = self.predict(state)
                latent_predictions["tr"][domain_name_target] = predictions[domain_name_target]

                masked_latents = {
                    domain_name: ([latents[domain_name][k]
                                   for k in range(len(latents[domain_name]))]
                                  if domain_name == domain_name_target
                                  else [null_latents[domain_name][k]
                                        for k in range(len(latents[domain_name]))])
                    for domain_name in latents.keys()
                }
                states[domain_name_target] = self.combine(self.project(masked_latents))

        demi_cycle_losses = self.loss(latent_predictions['dcy'], latent_targets, prefix="demi_cycles")
        cycle_losses = self.loss(latent_predictions['cy'], latent_targets, prefix="cycles")
        # cosine_losses = self.cosine_loss(states)
        contrastive_losses = self.contrastive_loss(states, masks)

        translation_losses = self.loss(latent_predictions['tr'], latent_targets, prefix="translation")

        losses = {**demi_cycle_losses, **cycle_losses, **translation_losses, **contrastive_losses}
        loss_names = ["demi_cycles", "cycles", "translation", "contrastive"]
        losses["total"] = torch.stack([self.hparams[f"loss_coef_{loss_name}"] * losses[loss_name]
                                       for loss_name in loss_names], dim=0).sum()
        losses["total_no_coefs"] = torch.stack([losses[loss_name] for loss_name in loss_names], dim=0).sum()

        batch_size = latents[list(latents.keys())[0]][0].size(0)
        for name, loss in losses.items():
            loss_name = f"{mode}/{prefix}{name}_loss"
            if mode == "val" and self.trainer.global_step == 0:
                for logger in self.loggers:
                    logger.set_summary(loss_name, "min")
            self.log(loss_name, loss, logger=True,
                     add_dataloader_idx=False, batch_size=batch_size)

        if mode == "train":
            for coef_name in ["demi_cycles", "cycles", "translation", "cosine"]:
                self.log(f"loss_coef/{coef_name}", getattr(self, f"loss_coef_{coef_name}"), add_dataloader_idx=False,
                         batch_size=batch_size)

        return losses["total"], losses

    def training_step(self, domains, batch_idx):
        # remove the sync batch
        available_domains, domains = split_domains_available_domains(domains)

        latents = self.encode_uni_modal(domains)

        total_loss, losses = self.step(available_domains, latents, mode="train")

        return total_loss

    def validation_step(self, domains, batch_idx, dataset_idx=0):

        available_domains, domains = split_domains_available_domains(domains)

        latents = self.encode_uni_modal(domains)
        prefix = "in_dist/" if dataset_idx == 0 else "ood/"
        total_loss, losses = self.step(available_domains, latents, mode="val", prefix=prefix)

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

        available_domains, domains = split_domains_available_domains(domains)

        latents = self.encode_uni_modal(domains)
        prefix = "in_dist/" if dataset_idx == 0 else "ood/"
        total_loss, losses = self.step(available_domains, latents, mode="test", prefix=prefix)
        return total_loss

    def get_domain_examples(self, set_name, dist):
        domain_examples = {}
        for domain_name, n_items in self.validation_example_list.items():
            domain_example = [
                getattr(self, f"{set_name}_{dist}_examples_domain_{domain_name}_{k}") for k in range(n_items)
            ]
            if len(domain_example) == 1:
                domain_example = domain_example[0]
            domain_examples[domain_name] = domain_example
        return domain_examples

    def log_images(self, logger, examples, slug="val", max_examples=None):
        available_domains, examples = split_domains_available_domains(examples)
        if self.current_epoch == 0:
            save_images = logger._save_images
            logger.save_images(True)
            for domain_name, domain_example in examples.items():
                self.domain_mods[domain_name].log_domain(logger, domain_example,
                                                         f"{slug}/original/domain_{domain_name}", max_examples)
            logger.save_images(save_images)
            # if domain_name == "v":
            #     latent = self.domain_mods[domain_name].encode(domain_example)[1].detach().cpu().numpy()
            #     fig, axes = plt.subplots(1, latent.shape[1])
            #     for k in range(latent.shape[1]):
            #         l = latent[:, k]
            #         axes[k].hist(l, 50, density=True)
            #         x = np.linspace(-0.8, 0.8, 100)
            #         axes[k].plot(x, scipy.stats.norm.pdf(x, 0, 1))
            #     logger.log_image("original_v_hist", fig)
            #     plt.close(fig)

        # if len(self.rotation_error_val):
        #     fig = plt.figure()
        #     plt.hist(self.rotation_error_val, 50, density=True)
        #     logger.log_image("rotation_error_val", fig)
        #     plt.close(fig)
        #     self.rotation_error_val = []

        latents = self.encode_uni_modal(examples)
        null_latents = {domain_name: self.get_null_latent(available_domains["v"].size(0), domain_name, True)
                        for domain_name in available_domains.keys()}

        for domain_name, latent in latents.items():
            # Demi cycles
            masked_latents = {
                domain_name_target: ([latents[domain_name_target][k] for k in range(len(latents[domain_name_target]))]
                                     if domain_name != domain_name_target
                                     else [null_latents[domain_name_target][k]
                                           for k in range(len(latents[domain_name_target]))])
                for domain_name_target in latents.keys()
            }
            predictions = self.adapt(self.predict(self.combine(self.project(masked_latents))))
            x_reconstructed = self.domain_mods[domain_name].decode(predictions[domain_name])
            self.domain_mods[domain_name].log_domain(logger, x_reconstructed,
                                                     f"{slug}/demi_cycles/{domain_name}", max_examples)

            for domain_name_2 in latents.keys():
                if domain_name_2 != domain_name:
                    masked_predictions = {
                        domain_name_target: (
                            [predictions[domain_name_target][k] for k in range(len(predictions[domain_name_target]))]
                            if domain_name_2 != domain_name_target
                            else [null_latents[domain_name_target][k]
                                  for k in range(len(latents[domain_name_target]))])
                        for domain_name_target in latents.keys()
                    }
                    # Full cycles
                    cycle_predictions = self.predict(self.combine(self.project(masked_predictions)))
                    x_reconstructed = self.domain_mods[domain_name].decode(cycle_predictions[domain_name])
                    self.domain_mods[domain_name].log_domain(logger, x_reconstructed,
                                                             f"{slug}/cycles/{domain_name}_through_{domain_name_2}",
                                                             max_examples)

                    # Translations
                    domain_end_pred = self.domain_mods[domain_name_2].decode(predictions[domain_name_2])
                    self.domain_mods[domain_name_2].log_domain(
                        logger, domain_end_pred,
                        f"{slug}/translation/{domain_name}_to_{domain_name_2}",
                        max_examples,
                    )
                    # if domain_name == "t" and domain_name_2 == "v":
                    #     fig, axes = plt.subplots(1, latent_end.size(1))
                    #     for k in range(latent_end.size(1)):
                    #         l = latent_end.detach().cpu().numpy()[:, k]
                    #         x = np.linspace(-0.8, 0.8, 100)
                    #         axes[k].hist(l, 50, density=True)
                    #         axes[k].plot(x, scipy.stats.norm.pdf(x, 0, 1))
                    #     logger.log_image("decoded_v_hist", fig)
                    #     plt.close(fig)

    def epoch_end(self, mode="val", log_train=True):
        domain_examples = self.domain_examples[mode]
        if self.domain_examples is not None:
            for logger in self.loggers:
                self.set_unimodal_pass_through(False)
                if self.current_epoch == 0:
                    if self.trainer.datamodule.ood_boundaries is not None:
                        logger.log_hyperparams({"ood_boundaries": self.trainer.datamodule.ood_boundaries})
                # self.logger.experiment["grad_norm_array"].upload(File.as_html(self.grad_norms_bin.values(15)))
                for k, dist in enumerate(["in_dist", "ood"]):
                    if domain_examples[k] is not None:
                        validation_examples = self.get_domain_examples(mode, dist)

                        if self.validation_example_list is not None:
                            self.log_images(logger, validation_examples, f"{mode}/{dist}")

                if log_train and "train" in self.domain_examples and self.domain_examples["train"][0] is not None:
                    train_examples = self.get_domain_examples("train", "in_dist")
                    self.log_images(logger, train_examples, "train")

                self.set_unimodal_pass_through(True)

    def validation_epoch_end(self, outputs):
        self.epoch_end("val", log_train=True)

    def test_epoch_end(self, outputs):
        self.epoch_end("test", log_train=False)

    def set_unimodal_pass_through(self, mode=True):
        for domain_mod in self.domain_mods.values():
            if isinstance(domain_mod, PassThroughWM):
                domain_mod.pass_through(mode)

    def on_train_epoch_start(self):
        self.domain_mods.eval()
        self.schedule_loss_coef_step()

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
            size_dataset = len(self.trainer.datamodule.shapes_train["sync_"])
            batch_size = self.trainer.datamodule.batch_size
            if scheduler_interval == "step":
                n_step_per_epoch = int(size_dataset / batch_size)
                scheduler_step /= n_step_per_epoch
            # If less data, we need to do more scheduler steps. Must depend on the synchronised data
            prop_labelled_images = 1. - self.trainer.datamodule.prop_sync_domains["all"]
            steps_per_new_epoch = int(scheduler_step * (size_dataset * prop_labelled_images) / batch_size)
            scheduler_step = max(1, steps_per_new_epoch)
            scheduler_interval = "step"
            print(f"Scheduler will be updated every {scheduler_step} step(s).")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_step, scheduler_gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
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
