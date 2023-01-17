from typing import Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT

from bim_gw.modules.ations import ActionModule
from bim_gw.modules.workspace_encoders import DomainDecoder, DomainEncoder


class WorldModel(LightningModule):
    def __init__(self, global_workspace, action_modality: ActionModule,
                 optimizer_lr=1e-3, optimizer_weight_decay=1e-5, scheduler_step=100, scheduler_gamma=0.1):
        super(WorldModel, self).__init__()
        self.save_hyperparameters()


        self.global_workspace = global_workspace
        self.z_size = self.global_workspace.z_size
        self.hidden_size = self.global_workspace.hidden_size
        self.action_modality = action_modality

        in_dims = [self.z_size] + self.action_modality.output_dims
        self.future_encoder = DomainEncoder(in_dims, self.hidden_size, self.z_size)
        self.past_encoder = DomainEncoder(in_dims, self.hidden_size, self.z_size)

        self.action_decoder = DomainDecoder(2 * self.z_size,
                                            self.hidden_size,
                                            self.action_modality.output_dims,
                                            self.action_modality.decoder_activation_fn)

    def predict_future(self, state, action):
        return self.future_encoder([state] + action)

    def predict_past(self, state, action):
        return self.past_encoder([state] + action)

    def predict_action(self, state_past, state_future):
        states = torch.cat([state_past, state_future], dim=1)
        return self.action_decoder(states)

    def forward(self, past_state=None, future_state=None, action=None):
        if past_state is not None and future_state is not None:
            return self.predict_action(past_state, future_state)
        elif past_state is not None and action is not None:
            return self.predict_future(past_state, action)
        elif future_state is not None and action is not None:
            return self.predict_past(future_state, action)
        else:
            raise ValueError("There is not enough information to predict anything.")

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pass

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hparams.optimizer_lr, weight_decay=self.hparams.optimizer_weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.hparams.scheduler_step,
                                                    self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]
