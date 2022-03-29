import math

import torch


class Memory(torch.nn.Module):
    """
    This is the base class for a memory cell.
    """

    def __init__(self, num_slots, memory_size):
        super(Memory, self).__init__()

        self.num_slots = num_slots
        self.memory_size = memory_size

        self.register_buffer("memory", torch.empty(self.num_slots, self.memory_size))

    def put(self, keys, values):
        self.memory[keys] = values

    def get(self, keys):
        return self.memory[keys]

    def forward(self, keys):
        return self.get(keys)


class MemoryKeyVector(Memory):
    """
    This is the base class for a memory cell.
    """

    def __init__(self, num_slots, key_size, memory_size):
        super(MemoryKeyVector, self).__init__(num_slots, memory_size)

        self.key_size = key_size
        # Use positional encoding to encode keys... Could be learned
        memory_slots = torch.arange(self.num_slots).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.key_size, 2) * (-math.log(10000.0) / self.key_size))
        memory_keys = torch.zeros(self.num_slots, self.key_size)
        memory_keys[:, 0::2] = torch.sin(memory_slots * div_term)
        memory_keys[:, 1::2] = torch.cos(memory_slots * div_term)
        self.register_buffer("memory_keys", memory_keys)

    def put(self, keys, values):
        self.memory[self.closest(keys).indices] = values

    def get(self, keys):
        return self.memory[self.closest(keys).indices]

    def closest(self, keys):
        distances = torch.matmul(keys, self.memory_keys.t())
        return torch.max(distances, dim=1)


if __name__ == '__main__':
    memory = MemoryKeyVector(10, 4, 12)
    val = torch.arange(12).unsqueeze(0).to(torch.float)
    key = torch.randn(1, 4)
    print(memory.closest(key))
    print(memory.get(key))
    memory.put(key, val)
    print(memory.get(key))
