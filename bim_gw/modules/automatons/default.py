import torch.nn
import torch.nn.functional as F


class Automaton(torch.nn.Module):
    """
    This is the base class for Automatons.
    Automatons decide of the operations made for computation.
    """

    def __init__(self, memory, n_modalities):
        super(Automaton, self).__init__()
        # Here we use a simple dict memory.
        # slot 0, 1, 2 is used for aux computation
        # slot 3 to n_modality + 2 contain the modalities at the beginning.

        self.memory = memory
        self.n_modalities = n_modalities  # should include action and past / future

        self.operations = {
            "NO_OP": 0,
            "PROJECT": 1,
            "COMBINE": 2,
            "PREDICT": 3
        }
        self.operation_size = len(self.operations) + n_modalities + self.memory.num_slots + self.memory.num_slots

    def get_operation(self, operation_name, batch_size):
        return F.one_hot(torch.full(batch_size, self.operations[operation_name], dtype=torch.long),
                         len(self.operations))

    def get_no_op_operation(self, batch_size):
        return [
            self.get_operation("NO_OP", batch_size),
            torch.zeros(batch_size, self.n_modalities).to(torch.long),
            torch.zeros(batch_size, self.memory.num_slots).to(torch.long),
            torch.zeros(batch_size, self.memory.num_slots).to(torch.long)
        ]

    def get_project_operation(self, modalities, memory_slot_in, memory_slot_out):
        return [
            self.get_operation("PROJECT", modalities.size(0)),
            modalities,
            memory_slot_in,
            memory_slot_out
        ]

    def get_combine_operation(self, memory_slot_in, memory_slot_out):
        return [
            self.get_operation("COMBINE", memory_slot_in.size(0)),
            torch.zeros(memory_slot_in.size(0), self.n_modalities),
            memory_slot_in,
            memory_slot_out
        ]

    def get_predict_operation(self, modality, memory_slot_in, memory_slot_out):
        return [
            self.get_operation("PREDICT", modality.size(0)),
            modality,
            memory_slot_in,
            memory_slot_out
        ]

    def get_operations_atemporal(self, modalities):
        """
        Args:
            modalities: tensor of size (n_batch, n_modalities)
        Returns:
        """
        # Start by adding all modalities
        operations = []
        for k in range(self.n_modalities):
            # Set all to no_op
            op = self.get_no_op_operation(modalities.size(0))
            # If modality is present, project it into the memory slot 0
            op[modalities[:, k] != 0] = self.get_project_operation(
                torch.full(modalities.size(0), k),
                torch.full(modalities.size(0), k + 3),
                torch.zeros(modalities.size(0))
            )
            operations.append(op)
        # App combined state in memory slot 1
        operations.append(
            self.get_combine_operation(
                torch.zeros(modalities.size(0)),
                torch.full(modalities.size(0), 1)
            )
        )
        # Add predict operation for all modalities
        for k in range(self.n_modalities):
            op = self.get_predict_operation(
                torch.full(modalities.size(0), k),
                torch.full(modalities.size(0), k + self.n_modalities + 3),
                torch.zeros(modalities.size(2))
            )
            operations.append(op)
        return operations

    def forward(self, modalities):
        """
        Args:
            modalities: tensor of size (n_batch, n_time_steps, n_modalities). n_modalities contains normal
                domains, actions, past_state, future_state
        Returns: List of operations
        """
        pass
