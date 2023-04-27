from collections import defaultdict
from collections.abc import Collection
from typing import Dict, Iterable

import torch
from torch.utils.data.dataloader import default_collate

from bim_gw.utils.types import AvailableDomains


# Can't use the collections.abc.MutableMapping interface because torch's
# pin_memory transforms it to a dict
# https://github.com/pytorch/pytorch/issues/70158
class DomainItems(Collection):
    def __init__(self, available_masks, **sub_parts):
        self.available_masks = available_masks
        self.sub_parts = sub_parts

    @staticmethod
    def singular(is_available=True, **sub_parts):
        return DomainItems(
            torch.tensor(is_available),
            **sub_parts,
        )

    def is_batch(self):
        return len(self) > 1

    def __getitem__(self, item):
        return self.sub_parts[item]

    def __setitem__(self, key, value):
        self.sub_parts[key] = value

    def __delitem__(self, key):
        del self.sub_parts[key]

    def __len__(self):
        return self.available_masks.shape[0]

    def __iter__(self):
        return iter(self.sub_parts)

    @staticmethod
    def get_collate_fn():
        return domain_collate_fn

    def to_device(self, device):
        self.available_masks = self.available_masks.to(device)
        self.sub_parts = _to_device(self.sub_parts, device)
        return self

    def pin_memory(self):
        self.available_masks = self.available_masks.pin_memory()
        self.sub_parts = _pin_memory(self.sub_parts)
        return self

    # Reimplementation of the MutableMapping interface
    def __contains__(self, key):
        return key in self.sub_parts

    def values(self):
        yield from self.sub_parts.values()

    def keys(self):
        yield from self.sub_parts.keys()

    def items(self):
        yield from self.sub_parts.items()


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    return value


def _pin_memory(value):
    if hasattr(value, "pin_memory"):
        return value.pin_memory()
    if isinstance(value, dict):
        return {k: _pin_memory(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_pin_memory(v) for v in value]
    return value


def domain_collate_fn(batch: Iterable[DomainItems]) -> DomainItems:
    items = [(item.available_masks, item.sub_parts) for item in batch]
    batched_items = default_collate(items)
    return DomainItems(batched_items[0], **batched_items[1])


def collate_fn(
    batch: Iterable[Dict[AvailableDomains, DomainItems]]
) -> Dict[AvailableDomains, DomainItems]:
    items = defaultdict(list)
    for item in batch:
        for domain_name, domain_item in item.items():
            items[domain_name].append(domain_item)
    return {
        domain_name: domain_collate_fn(items[domain_name])
        for domain_name in items.keys()
    }
