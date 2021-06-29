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


class GlobalWorkspace(LightningModule):
    def __init__(self, domain_mods, z_size,
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

    def translate(self, x, domain1, domain2):
        """
        Translates x from domain1 to domain2
        """
        z = self.encoders[domain1](x)
        return self.decoders[domain2](z)

    def demi_cycle(self, x, domain):
        return self.translate(x, domain, domain)

    def cycle(self, x, domain1, domain2):
        z = self.translate(x, domain1, domain2)
        return self.translate(z, domain2, domain1)

    def training_step(self, batch, batch_idx):
        domains = {"v": batch[0], "t": batch[1]}
        out = self(domains)

    def validation_step(self, batch, batch_idx):
        domains = {"v": batch[0], "t": batch[1]}
        out = self(domains)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optim_lr,
                                     weight_decay=self.hparams.optim_weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.scheduler_step,
                                                    self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]
