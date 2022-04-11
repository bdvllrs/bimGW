import random
from typing import Iterator

from torch.utils.data import Sampler as BaseSampler
from torch.utils.data.sampler import T_co


class Sampler(BaseSampler):
    def __init__(self, batch_size, domain_map):
        """
        Args:
            batch_size:
            domain_map: of the form: {"['t', 'v']": [0, 2, ...], "['t']": [1, ...], ...}
        """
        self.batch_size = batch_size
        self.domain_map = domain_map
        self.n_items = sum([len(x) for x in self.domain_map.values()])
        self.seen_items = {key: set() for key in self.domain_map.keys()}

    def __len__(self):
        return self.n_items // self.batch_size

    def __iter__(self) -> Iterator[T_co]:
        self.seen_items = {key: set() for key in self.domain_map.keys()}
        for k in range(self.n_items // self.batch_size):
            batch = []
            for i, (key, possible_keys) in enumerate(self.domain_map.items()):
                intersected_items = self.seen_items[key].intersection(possible_keys)
                if len(intersected_items):
                    if i == len(self.domain_map.keys()) - 1:
                        n_items = self.batch_size - len(batch)
                    else:
                        n_items = self.batch_size // len(self.domain_map.keys())
                    batch.extend(random.sample(list(intersected_items), n_items))
                    self.seen_items[key] = self.seen_items[key].union(set(batch))
            yield batch