import copy
from collections.abc import Collection

import torch
from torch.utils.data.dataloader import default_collate


# Can't use the collections.abc.MutableMapping interface because torch's
# pin_memory transforms it to a dict
# https://github.com/pytorch/pytorch/issues/70158
class DomainItems(Collection):
    def __init__(self, available_masks, **sub_parts):
        self.available_masks = available_masks
        self._sub_parts = sub_parts

    @staticmethod
    def singular(is_available=True, **sub_parts):
        return DomainItems(
            torch.tensor(is_available),
            **sub_parts,
        )

    def is_batch(self):
        return len(self) > 1

    def _get_attr(self, key):
        return self._sub_parts[key]

    def __getattr__(self, key):
        try:
            return self._get_attr(key)
        except KeyError:
            raise AttributeError(f"DomainItems has no attribute {key}.")

    def __getitem__(self, item):
        return self._get_attr(item)

    def __setitem__(self, key, value):
        self._sub_parts[key] = value

    def __delitem__(self, key):
        del self._sub_parts[key]

    def __len__(self):
        return self.available_masks.shape[0]

    def __iter__(self):
        return iter(self._sub_parts)

    @property
    def sub_parts(self):
        return self._sub_parts

    @staticmethod
    def get_collate_fn():
        return domain_collate_fn

    def to_device(self, device):
        self.available_masks = self.available_masks.to(device)
        self._sub_parts = _to_device(self.sub_parts, device)
        return self

    def pin_memory(self):
        self.available_masks = self.available_masks.pin_memory()
        self._sub_parts = _pin_memory(self.sub_parts)
        return self

    # Reimplementation of the MutableMapping interface
    def __contains__(self, key):
        return key in self._sub_parts

    def values(self):
        yield from self._sub_parts.values()

    def keys(self):
        yield from self._sub_parts.keys()

    def items(self):
        yield from self._sub_parts.items()


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        new_dict = {}
        for key, item in value.items():
            new_dict[key] = _to_device(item, device)
        return new_dict
    if isinstance(value, list):
        new_list = []
        for item in value:
            new_list.append(_to_device(item, device))
        return new_list
    return copy.copy(value)


def _pin_memory(value):
    if isinstance(value, torch.Tensor):
        return value.pin_memory()
    if isinstance(value, dict):
        new_dict = {}
        for key, item in value.items():
            new_dict[key] = _pin_memory(item)
        return new_dict
    if isinstance(value, list):
        new_list = []
        for item in value:
            new_list.append(_pin_memory(item))
        return new_list
    return copy.copy(value)


def domain_collate_fn(batch):
    items = [(item.available_masks, item.sub_parts) for item in batch]
    batched_items = default_collate(items)
    return DomainItems(batched_items[0], **batched_items[1])


def collate_fn(batch):
    items = {domain_name: [] for domain_name in batch[0].keys()}
    for item in batch:
        for domain_name, domain_item in item.items():
            items[domain_name].append(domain_item)
    return {
        domain_name: domain_collate_fn(items[domain_name])
        for domain_name in items.keys()
    }
