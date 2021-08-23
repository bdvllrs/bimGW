import torch
import torchmetrics
from omegaconf import ListConfig
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F

from bim_gw.utils.losses import vis_to_text_accuracy
from bim_gw.utils.shapes import log_shape_fig
from bim_gw.utils.utils import log_image, val_or_default


class EncoderBlock(torch.nn.Sequential):
    def __init__(self, in_dim, out_dim):
        super(EncoderBlock, self).__init__(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )


class DomainDecoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim, pose_dim=0, loss_fn=None):
        super(DomainDecoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.pose_dim = pose_dim
        self.loss_fn = loss_fn

        self.encoder_block1 = EncoderBlock(self.in_dim, self.hidden_size)
        self.encoder_block2 = EncoderBlock(self.hidden_size, self.hidden_size)
        self.encoder_block3 = nn.Sequential(
            nn.Linear(self.hidden_size, self.out_dim),
        )
        if self.pose_dim > 0:
            self.encoder_block_pose = nn.Sequential(
                nn.Linear(self.hidden_size, self.pose_dim),
            )

    def forward(self, x):
        out = self.encoder_block1(x)
        out = self.encoder_block2(out)
        z = None
        if self.pose_dim > 0:
            z = self.encoder_block_pose(out)
        out = self.encoder_block3(out)
        if self.loss_fn is not None:
            out = self.loss_fn(out)
            z = self.loss_fn(z)
        return out, z


class DomainEncoder(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim, pose_dim=0):
        super(DomainEncoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.pose_dim = pose_dim

        self.encoder_block1 = EncoderBlock(self.in_dim + self.pose_dim, self.hidden_size)
        self.encoder_block2 = EncoderBlock(self.hidden_size, self.hidden_size)
        self.encoder_block3 = nn.Sequential(
            nn.Linear(self.hidden_size, self.out_dim),
        )

    def forward(self, x, z=None):
        if self.pose_dim != 0 and z is None:
            raise ValueError("z should be provided.")
        if z is not None:
            x = torch.cat((x, z), dim=-1)
        out = self.encoder_block1(x)
        out = self.encoder_block2(out)
        out = self.encoder_block3(out)
        return torch.tanh(out)


class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_size):
        super(Discriminator, self).__init__()
        self.in_dim = in_dim
        self.hidden_size = hidden_size

        self.encoder_block1 = EncoderBlock(self.in_dim, self.hidden_size)
        self.encoder_block2 = EncoderBlock(self.hidden_size, self.hidden_size)
        self.encoder_block3 = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, x):
        out = self.encoder_block1(x)
        out = self.encoder_block2(out)
        out = self.encoder_block3(out)
        return torch.sigmoid(out)


class DiscriminatorDecoder(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim):
        super(DiscriminatorDecoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size

        self.decoder = DomainDecoder(in_dim, self.hidden_size, out_dim)
        self.discriminator = Discriminator(out_dim, self.hidden_size)

    def forward(self, x):
        return self.decoder(x)


def check_domains_eq(domains_ori, domains_comp):
    for o, r in zip(domains_ori.values(), domains_comp.values()):
        assert torch.eq(o[0], r[0]).all()
        if o[1] is not None:
            assert torch.eq(o[1], r[1]).all()


def cross_entropy(x, y):
    y = torch.argmax(y, 1)
    return F.cross_entropy(x, y)


loss_functions = {
    "cosine": lambda x, y: -F.cosine_similarity(x, y),
    "mse": F.mse_loss,
    "cross_entropy": cross_entropy
}


class GlobalWorkspace(LightningModule):
    def __init__(
            self, domain_mods, z_size, hidden_size,
            n_classes=1000,
            loss_coef_demi_cycles=1, loss_coef_cycles=1, loss_coef_supervision=1,
            cycle_loss_fn="cosine", supervision_loss_fn=None,
            optim_lr=3e-4, optim_weight_decay=1e-5, scheduler_step=20, scheduler_gamma=0.5,
            pose_noise_dim=None,
            n_validation_examples: int = 32,
            validation_reconstructed_images=None,
            validation_reconstructed_targets=None,
            monitor_grad_norms=False
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

        self.pose_noise_dim = pose_noise_dim if pose_noise_dim is not None else dict()

        # Define encoders for translation
        self.encoders = nn.ModuleDict({item: DomainEncoder(mod.z_size, self.hidden_size, self.z_size,
                                                           (pose_noise_dim[item]
                                                            if item in pose_noise_dim else 0))
                                       for item, mod in domain_mods.items()})
        self.decoders = nn.ModuleDict({item: (DomainDecoder(self.z_size, self.hidden_size,
                                                            mod.z_size, (pose_noise_dim[item]
                                                                         if item in pose_noise_dim else 0),
                                                            None if item != "t" else torch.sigmoid))
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

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

        # val sampling
        self.register_buffer("validation_reconstruction_images", validation_reconstructed_images)
        self.register_buffer("validation_reconstruction_targets", validation_reconstructed_targets)
        self.register_buffer("validation_class_translation",
                             torch.randint(0, n_classes, (n_validation_examples,)).to(torch.int64))
        self.register_buffer("validation_pose_translation",
                             domain_mods['t'].get_random_vector(self.validation_class_translation))
        if "t" in self.pose_noise_dim:
            self.register_buffer("validation_pose_t",
                                 torch.randn(n_validation_examples, self.pose_noise_dim["t"]))
        else:
            self.register_buffer("validation_pose_t", None)

    def project(self, domains):
        """
        Projects unimodal domains to global workspace
        """
        out = dict()
        for domain_name, x in domains.items():
            c, z = self.domain_mods[domain_name].encode(x)
            # If nothing is returned but the domain still requires a latent vector, one is selected at random.
            if z is None and domain_name in self.pose_noise_dim:
                z = torch.randn(c.size(0), self.pose_noise_dim[domain_name]).type_as(c)
            out[domain_name] = (c, z)
        return out

    def forward(self, domains):
        pass

    def translate(self, x, domain_name_start, domain_name_target):
        """
        Translates x from domain1 to domain2
        """
        z = self.encoders[domain_name_start](*x)
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
            l = coef * self.cycle_loss_fn(domain[0], out[0]).mean() / total
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
                    l = coef * self.cycle_loss_fn(domain[0], out[0]).mean() / total
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

        accuracy = vis_to_text_accuracy(self, self.train_acc, sync_latents["v"], sync_supervision["t"])
        self.log("train_vis_to_text_acc", accuracy, on_step=True, on_epoch=False)

        if self.monitor_grad_norms:
            grad_norms = self.manual_backward_with_grad_norm_monitoring(losses)
            for name, grad_norm in grad_norms.items():
                self.log(f"grad_norm_{name}", grad_norm, logger=True)
        else:
            self.manual_backward(total_loss)

        opt.step()
        return total_loss

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

                        grad_norms[f"{name}_{param_group}"] = torch.tensor(0.).type_as(loss) + sum([
                            # remove the already saved gradient that have already been counted in.
                            (p.grad.detach() - val_or_default(last_grads, f"{param_group}_{param_name}", 0)).norm()
                            for param_name, p in model[modality].named_parameters()
                            if p.grad is not None
                        ])
                        assert grad_norms[f"{name}_{param_group}"] >= 0

                        # Keep track of the value of the gradients to avoid counting
                        # them multiple times because of accumulation.
                        for param_name, p in model[modality].named_parameters():
                            if param_name not in last_grads:
                                last_grads[f"{param_group}_{param_name}"] = 0
                            if p.grad is not None:
                                last_grads[f"{param_group}_{param_name}"] += p.grad.detach()
        return grad_norms

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

        accuracy = vis_to_text_accuracy(self, self.valid_acc, latents["v"], domains["t"])
        self.log("val_vis_to_text_acc", accuracy, on_step=True, on_epoch=True)

        return total_loss

    def validation_epoch_end(self, outputs):
        x = self.validation_reconstruction_images

        if self.current_epoch == 0:
            log_image(self.logger, x[:self.hparams.n_validation_examples], "val_original_images")
            classes = [self.trainer.datamodule.classes[k] for k in self.validation_class_translation]
            self.log("val_generated_labels", ", ".join(classes[:self.hparams.n_validation_examples]))
            classes = [self.trainer.datamodule.classes[k] for k in self.validation_reconstruction_targets]
            self.log("val_original_labels", ", ".join(classes[:self.hparams.n_validation_examples]))

            # Generate the image with the original (exact) algorithm
            t_gen = (self.validation_class_translation, self.validation_pose_translation)
            log_shape_fig(self.logger, t_gen[0].detach().cpu().numpy(),
                          t_gen[1].detach().cpu().numpy(), "val_generated_labels_vis")

        # translation v -> t
        latent_v = self.domain_mods["v"].encode((x, None))
        latent_t = self.translate(latent_v, "v", "t")
        # print(latent_v[0][0])
        # z = self.encoders["v"](*latent_v)
        # print(z[0])
        # latent_t = self.decoders["t"](z)
        t_gen = self.domain_mods["t"].decode(latent_t)
        # Get class
        predicted_classes = t_gen[0].detach().cpu().numpy()
        classes = [self.trainer.datamodule.classes[k] for k in predicted_classes]
        self.log("val_translation_v_to_t_text", ", ".join(classes[:self.hparams.n_validation_examples]))

        # Use the exact algo to generate the visualisation.
        log_shape_fig(
            self.logger,
            t_gen[0][:self.hparams.n_validation_examples].detach().cpu().numpy(),
            t_gen[1][:self.hparams.n_validation_examples].detach().cpu().numpy(),
            "val_translation_v_to_t_vis"
        )

        # Translation t -> v
        latent_t = self.domain_mods["t"].encode((self.validation_class_translation, self.validation_pose_translation))
        latent_v = self.translate(latent_t, "t", "v")
        v_gen = self.domain_mods["v"].decode(latent_v)[0]
        log_image(self.logger, v_gen[:self.hparams.n_validation_examples], "val_translation_t_to_v")

        # demi cycle v
        latent_x = self.domain_mods["v"].encode((x, None))
        latent_reconstructed = self.demi_cycle(latent_x, "v")
        x_reconstructed = self.domain_mods["v"].decode(latent_reconstructed)[0]
        log_image(self.logger, x_reconstructed[:self.hparams.n_validation_examples], "val_demi_cycle_v")

        # demi cycle t
        latent_t = self.domain_mods["t"].encode((self.validation_class_translation, self.validation_pose_translation))
        latent_reconstructed = self.demi_cycle(latent_t, "t")
        t_reconstructed = self.domain_mods["t"].decode(latent_reconstructed)
        log_shape_fig(
            self.logger,
            t_reconstructed[0][:self.hparams.n_validation_examples].detach().cpu().numpy(),
            t_reconstructed[1][:self.hparams.n_validation_examples].detach().cpu().numpy(),
            "val_demi_cycle_t"
        )

        # full cycle v
        latent_x = self.domain_mods["v"].encode((x, None))
        latent_reconstructed = self.cycle(latent_x, "v", "t")
        x_reconstructed = self.domain_mods["v"].decode(latent_reconstructed)[0]
        log_image(self.logger, x_reconstructed[:self.hparams.n_validation_examples], "val_cycle_v")

        # full cycle t
        latent_t = self.domain_mods["t"].encode((self.validation_class_translation, self.validation_pose_translation))
        latent_reconstructed = self.cycle(latent_t, "t", "v")
        t_reconstructed = self.domain_mods["t"].decode(latent_reconstructed)
        log_shape_fig(
            self.logger,
            t_reconstructed[0][:self.hparams.n_validation_examples].detach().cpu().numpy(),
            t_reconstructed[1][:self.hparams.n_validation_examples].detach().cpu().numpy(),
            "val_cycle_t"
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
