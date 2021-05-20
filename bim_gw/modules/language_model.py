import torch
from torch import nn
from pytorch_lightning import LightningModule
import gensim.models


class LanguageModel(LightningModule):
    def __init__(self, gensim_model_path, vocab_size, latent_dim, hidden_dim,
                 optim_lr=3e-4, optim_weight_decay=1e-5,
                 scheduler_step=20, scheduler_gamma=0.5):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        print("Loading word2vec model.")
        self.encoder = gensim.models.KeyedVectors.load_word2vec_format(gensim_model_path, binary=False)
        print("Loaded.")

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.vocab_size)
        )

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optim_lr,
                                     weight_decay=self.hparams.optim_weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.scheduler_step, self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]
