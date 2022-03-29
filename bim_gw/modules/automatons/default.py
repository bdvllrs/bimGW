import torch.nn
import torch.nn.functional as F


class Automaton(torch.nn.Module):
    """
    This is the base class for Automatons.
    Automatons decide of the operations made for computation.
    """
    def __init__(self, memory, n_modalities):
        super(Automaton, self).__init__()
        self.memory = memory
        self.n_modalities = n_modalities  # should include action and past / future

        self.operations = {
            "PROJECT": 0,
            "COMBINE": 1,
            "PREDICT": 2
        }
        self.operation_size = len(self.operations) + n_modalities + self.memory.num_slots + self.memory.num_slots

    def get_operation(self, operation_name, batch_size):
        return F.one_hot(torch.full(batch_size, self.operations[operation_name], dtype=torch.long),
                         len(self.operations))

    def get_project_operation(self, modalities, memory_slot_in, memory_slot_out):
        return torch.cat([
            self.get_operation("PROJECT", modalities.size(0)),
            modalities,
            memory_slot_in,
            memory_slot_out
        ])

    def get_combine_operation(self, memory_slot_in, memory_slot_out):
        return torch.cat([
            self.get_operation("COMBINE", memory_slot_in.size(0)),
            torch.zeros(memory_slot_in.size(0), self.n_modalities),
            memory_slot_in,
            memory_slot_out
        ])

    def get_predict_operation(self, modality, memory_slot_in, memory_slot_out):
        return torch.cat([
            self.get_operation("PREDICT", modality.size(0)),
            modality,
            memory_slot_in,
            memory_slot_out
        ])

    def get_operations_atemporal(self, modalities):
        operations = torch.tensor(modalities.size(0), self.operation_size)

    def forward(self, modalities):
        """
        Args:
            modalities: tensor of size (n_batch, n_time_steps, n_modalities). n_modalities contains normal
                domains, actions, past_state, future_state
        Returns: List of operations
        """
        pass