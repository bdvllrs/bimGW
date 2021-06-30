import torch
from pytorch_lightning import LightningModule
from torch import nn


class DomainEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DomainEncoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.encoder = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        return self.encoder(x)


def check_domains_eq(domains_ori, domains_comp):
    for o, r in zip(domains_ori.values(), domains_comp.values()):
        assert torch.eq(o, r).all()


class GlobalWorkspace(LightningModule):
    def __init__(self, domain_mods, z_size,
                 loss_coef_demi_cycles=1, loss_coef_cycles=1, loss_coef_supervision=1,
                 optim_lr=3e-4, optim_weight_decay=1e-5, scheduler_step=20, scheduler_gamma=0.5):
        super(GlobalWorkspace, self).__init__()
        self.save_hyperparameters()

        self.z_size = z_size

        for mod in domain_mods.values():
            assert hasattr(mod, "z_size"), "Module must have a parameter z_size."

        self.domain_mods = nn.ModuleDict(domain_mods)

        # Define encoders for translation
        self.encoders = nn.ModuleDict({item: DomainEncoder(mod.z_size, self.z_size)
                                       for item, mod in domain_mods.items()})
        self.decoders = nn.ModuleDict({item: DomainEncoder(self.z_size, mod.z_size)
                                       for item, mod in domain_mods.items()})

        # cosine distance.
        self.cycle_loss_fn = lambda x, y: 1 - torch.nn.functional.cosine_similarity(x, y)
        self.supervision_loss_fn = self.cycle_loss_fn

    def project(self, domains):
        """
        Projects unimodal domains to global workspace
        """
        out = dict()
        for domain_name, x in domains.items():
            # returns a tuple of data. The first element is always the latent vector.
            domain_output = self.domain_mods[domain_name](x)
            assert type(domain_output) == tuple, "The forward method of modules should return tuples."
            out[domain_name] = domain_output[0]
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

    def demi_cycle_loss(self, domains):
        loss = torch.tensor(0.).to(self.device)
        for name, domain in domains.items():
            out = self.demi_cycle(domain, name)
            loss += self.cycle_loss_fn(domain, out).mean()
        return loss / len(domains)

    def cycle_loss(self, domains):
        loss = torch.tensor(0.).to(self.device)
        for domain_name_start, domain in domains.items():
            for domain_name_inter in domains.keys():
                if domain_name_start != domain_name_inter:
                    out = self.cycle(domain, domain_name_start, domain_name_inter)
                    loss += self.cycle_loss_fn(domain, out).mean()
        n = len(domains)
        return loss / (n * (n - 1))

    def supervision_loss(self, sync_domains):
        loss = torch.tensor(0.).to(self.device)
        for domain_name_1, domain_1 in sync_domains.items():
            for domain_name_2, domain_2 in sync_domains.items():
                if domain_name_1 != domain_name_2:
                    # project domains into the same multi modal workspace
                    latent_domain_1 = self.encoders[domain_name_1](domain_1)
                    latent_domain_2 = self.encoders[domain_name_2](domain_2)
                    loss += self.supervision_loss_fn(latent_domain_1, latent_domain_2).mean()
        n = len(sync_domains)
        return loss / (n * (n - 1))

    def training_step(self, batch, batch_idx):
        # remove the sync batch
        domains = {key: val for key, val in batch.items() if key != "sync_"}
        sync_supervision = batch["sync_"]  # Sparse cross-modal supervision
        latents = self.project(domains)
        ori_latents = latents.copy()
        demi_cycle_loss = self.hparams.loss_coef_demi_cycles * self.demi_cycle_loss(latents)

        check_domains_eq(ori_latents, latents)
        cycle_loss = self.hparams.loss_coef_cycles * self.cycle_loss(latents)
        check_domains_eq(ori_latents, latents)
        sync_latents = self.project(sync_supervision)
        supervision_loss = self.hparams.loss_coef_cycles * self.supervision_loss(sync_latents)

        total_loss = demi_cycle_loss + cycle_loss + supervision_loss

        self.log("train_demi_cycle_loss", demi_cycle_loss, logger=True)
        self.log("train_cycle_loss", cycle_loss, logger=True)
        self.log("train_supervision_loss", supervision_loss, logger=True)
        self.log("train_total_loss", total_loss, logger=True)

        return total_loss

    def validation_step(self, domains, batch_idx):
        latents = self.project(domains)
        ori_latents = latents.copy()
        demi_cycle_loss = self.hparams.loss_coef_demi_cycles * self.demi_cycle_loss(latents)

        check_domains_eq(ori_latents, latents)
        cycle_loss = self.hparams.loss_coef_cycles * self.cycle_loss(latents)
        check_domains_eq(ori_latents, latents)
        supervision_loss = self.hparams.loss_coef_cycles * self.supervision_loss(latents)

        total_loss = demi_cycle_loss + cycle_loss + supervision_loss

        self.log("val_demi_cycle_loss", demi_cycle_loss, logger=True)
        self.log("val_cycle_loss", cycle_loss, logger=True)
        self.log("val_supervision_loss", supervision_loss, logger=True)
        self.log("val_total_loss", total_loss, logger=True)

        return total_loss

    def on_train_epoch_start(self):
        self.domain_mods.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optim_lr,
                                     weight_decay=self.hparams.optim_weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.scheduler_step,
                                                    self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]

