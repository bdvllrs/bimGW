import torch

from bim_gw.modules.ations import ActionModule
from bim_gw.modules.utils import DomainDecoder, DomainEncoder


class WorldModel(torch.nn.Module):
    def __init__(self, z_size, hidden_size, action_modality: ActionModule):
        super(WorldModel, self).__init__()

        self.z_size = z_size
        self.hidden_size = hidden_size
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
